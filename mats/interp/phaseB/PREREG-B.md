# Phase B pre-registration: noisy-channel formulation (written 2026-10-28 ~16:05Z)

Before any noisy (sigma > 0) training result and before any q fit. Only these had been run when this was written:
- the sigma = 0 fixed-norm fine-tunes of both models (plumbing: does the architecture change recover near-Bayes);
- the linear-Gaussian estimator validation of task 1 (started, not yet read).

Code: `nnoise.py` (noisy fixed-norm PFN, fine-tune, eval), `mi.py` (mixture estimators), `task1.py` (random nets),
`fam.py` / `task3.py` (families; written after this file, implementing exactly the spec below).

## Architecture (exact)
Every written activation is projected to the sphere ||s|| = sqrt(dim) (no learned gain), then N(0, sigma^2 I) is added.
PFN noisy sites per token: emb (after input projection), a_l (residual after attention l), m_l (MLP hidden, 64 wide),
r_l (residual after MLP l). Readers (LN1, LN2, final LN) unchanged. Per-token power is exactly 1 per coordinate, so
- per token: I <= (dim/2) log2(1 + 1/sigma^2) bits (dim = 32 residual, 64 MLP hidden);
- whole layer (16 tokens): 16x that.
- Markov chain over residual cuts X -> emb -> a1 -> r1 -> a2 -> r2 (-> a3 -> r3); m_l hangs off a_l.
Random net (task 1): LayerNorm without affine after ReLU, so s lies in the (d-1)-dim centred subspace with ||s||^2 = d:
I(X;H_l) <= ((d-1)/2) log2(1 + d/((d-1) sigma^2)).

## Estimators (all bits)
p_l is exactly E_parent N(h; mu(parent), sigma^2 I). With a bank of M parents: leave-in (own parent included) gives
E log p_in >= E log p (InfoNCE identity) and leave-out gives E log p_out <= E log p (Jensen). So h(p) is bracketed,
and with K inner parent chains per x the same holds for h(H|X); I(X;H) in [h_in - hc_out, h_out - hc_in].
The lower end cannot exceed log2 M. A bracket is called *reliable* if its width is <= 1 bit.
- KL(p||q) = CE - h(p), CE = -E_p log q exact for tractable q: bracketed by the h(p) bracket.
- KL(q||p) (wasted bits): -h(q) - E_q log p_out; an over-estimate (Jensen), reported as such.
- bits_rate(q) = h(q) - h(sigma Z) (what q expresses at noise precision), exact up to MC.
- bits_desc(q) = H(K) + E_k sum 0.5 log2 max(lambda/sigma^2, 1) over the eigenvalues of component k's covariance
  (Kelvin's formula: discrete code entropy + Gaussian-channel content of the continuous part).
  bits_desc - bits_rate >= 0 measures code distinctions that the noise destroys.
- efficiency = I(X;H)/bits_desc (bracketed by the I bracket); also I/bits_rate.

## Task 2 runs
Warm start from the best clean checkpoints (deviation from "retrain from scratch", for CPU budget; the clean nets do
not survive the architecture change, gap ~3 nats, so the fine-tune is a real retraining of the function):
2L = tiny-pfn/runs/w32_h2_L2_mix4_ws_s4, 3L = tiny-pfn-interp/heads/hruns/L3_ws_s13.
sigma in {0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6}; sigma ramped 0 -> target over 1k steps, 20k steps lr 1e-3 then 8k cooldown;
best checkpoint by noise-marginalised val NLL (4 draws). Eval: 8,000 test episodes, 16 noise draws;
cost = KL(Bayes || noise-marginalised predictive) by quadrature, also mean single-draw KL.
MI: bank M = 2^15 episodes, N = 2048 targets, K = 32 inner chains, at every site for token groups ALL, QT, QQ0.

## Task 3 spec
Models: 2L and 3L at two sigmas: sigma_lo = largest sigma with cost - cost(0) <= 0.01 nats; sigma_hi = smallest sigma
with cost - cost(0) >= 0.05 nats (else the largest). Sites: token QT and QQ0 at every cut. Fit split 20k episodes,
p-bank M = 2^15, 4,096 eval targets.
Families (every Gaussian covariance floored at sigma^2 I):
1. **F1 named-Gaussian**: q(h) = (1/4096) sum_j N(h; A z_j + b, Sigma), z = fixed named world latents of the token
   (QT: Bayes mean, Bayes log var, 3 query readings, 3 log SS; QQ_s: own Q_s, own log SS_s, labelled-only posterior
   over v_s (3)), bank of 4,096 world draws; ridge mean, full residual covariance.
2. **F2 grid**: top-3 named latents by probe R^2 (fit split), 4 quantile levels each, 64 cells, Gaussian per cell
   (full covariance shrunk to the pooled one with weight n/(n+50)).
3. **F3 activation patterns** (small test: QT sites of 2L): cells = sign pattern of the 64 MLP pre-activations of the
   MLP reading the site (a_l) or writing it (m_l, r_l); per-pattern mean, shared covariance; patterns with < 5 fit
   samples pooled into one component. bits_desc charges the plug-in pattern entropy.
4. **F4 sparse**: TopK SAE, F = 4 x dim features, k in {2, 4}; q = mixture over observed supports S (weights pi_S),
   h|S ~ N(b + D_S m_S, D_S diag(v_S) D_S^T + s_e^2 I), s_e^2 = residual variance (>= sigma^2).
   bits_desc = H(S) + sum_active 0.5 log2(1 + |d_i|^2 v_i / s_e^2) + (dim/2) log2(s_e^2/sigma^2).
   Layer map: transcoder graph from the code at a_l to the code at r_l (token-local MLP sublayer): in-degree needed to
   reach 90% of the dense linear R^2 of each r_l feature's pre-activation.
- Chain consistency (token-local transition a_l -> r_l only; attention transitions need the other tokens):
  push q_{a_l} samples through the real noisy MLP sublayer; report KL(F#q_{a_l} || q_{r_l}) bracket.
  Total chain cost: sum over sites of bits_desc + eps.
- "Efficient" at a site: eps upper end <= 1 bit and efficiency (lower end) >= 0.5.

## Predictions
Task 1 (random nets, d in {8,16,32}, sigma in {0.1,0.3,1}, din in {2,8}, 3 layers)
- T1. Validation: the bracket contains the exact I whenever M >= 2^(I+3); the lower end never exceeds log2 M.
- T2. DPI holds within brackets in every reliable config: I(X;H1) >= I(X;H2) >= I(X;H3).
- T3. At sigma = 1, I(X;H3) <= 0.8 I(X;H1) in >= 2/3 of the reliable configs (most of the loss in layer 2 and 3).
- T4. Coarse-graining: k-means cells of E[H3|x] with 2^b cells capture >= 90% of I(X;H3) at b <= ceil(I(X;H3)) + 2.
- T5. Plug-in ReLU pattern entropy H(pattern_l) >= 2 I(X;H_l) at sigma >= 0.3 in >= 2/3 of configs (layers 2, 3).
Task 2
- P1. Cost (marginalised KL above the sigma=0 fine-tune) is monotone in sigma: <= 0.005 nats at sigma <= 0.1,
  >= 0.05 nats at sigma = 0.8; at sigma = 1.6 the 2L net is at or above the query-only plateau (KL >= 0.14).
- P2. At sigma >= 0.4 the 3L cost exceeds the 2L cost (more noise injections; same width).
- P3. ALL-token sites: the I lower end is within 1 bit of log2 M (saturated, unreliable) at every site for sigma <= 0.4.
- P4. QT token at the last residual cut: reliable bracket (<= 1 bit wide) for sigma >= 0.4.
- P5. Single-draw KL exceeds the marginalised KL by >= 20% at sigma >= 0.4.
Task 3
- P6. F3: bits_desc - bits_rate >= 0.5 bits_desc at every QT site tested; F3 has the lowest efficiency of the families at
  >= 75% of its sites.
- P7. At late QT cuts (a_L, r_L) F2 (named grid) is the most efficient family; at QQ0 sites F4 (sparse) is.
- P8. Chain: KL(F#q_{a_l} || q_{r_l}) <= eps(q_{r_l}) + 1 bit for F1 and F2; F3 fails this (> 2 bits) where tested.
- P9. At least one site has no efficient family at sigma_lo (candidate limit); expected: QQ0 at a1/r1, where the net
  carries ID-binding information that no named world latent describes.

---

## Amendment 1 (2026-10-28 ~16:45Z): retarget to the agreed program (`outbox/cm-interp-program-2026-10-28.md`)

Written after: task-1 flow results (random nets, read), the sigma = 0 / 0.4 fine-tunes (val gaps seen in logs:
sigma 0.4 costs ~0.10 nats, sigma 1.6 ~0.55), a smoke run of the task-3 code on an *unadapted* model. No task-3 result on
a noise-trained model and no deductive result of any kind exists yet.

**What stays.** Task 1 (random nets, I(X;H_l), DPI, coarse-graining) and task 2 (noisy training, cost curve, capacity,
MI) as specified above. The sample-based family fits of task 3 are kept only as a *validation* layer.
Added estimator (post hoc, stated as such): the x-level Gaussian-conditional estimator (`mi.xgauss_bracket`), which needs
M >> 2^I(X;H) instead of M >> 2^I(parent;H); it is an approximation (validated on linear-Gaussian chains and against the
exact bracket on width-8 random nets), not a bound.

**Architecture check (coordinator spec).** Already satisfied: noise sits after a fixed-norm projection at every site
(post-norm residual h = sphere(h + F(h)) + sigma*eps; MLP hidden = sphere(GELU(.)) + sigma*eps); noise stays on at
evaluation and in all analysis. Exact per-token bound: I <= (dim/2) log2(1 + 1/sigma^2), dim = 32 (residual) / 64 (MLP).

**Sigma rule (replaces sigma_lo / sigma_hi).** The sweep adds SNR 1000, 300, 10, 3 (sigma 0.0316, 0.0577, 0.316, 0.577).
Analysis sigma = the largest sigma with (KL to Bayes) - (KL of the sigma = 0 fixed-norm model) <= 0.01 nats AND mean
KL(noiseless model's predictive || noisy model's predictive) <= 0.01 nats. The full curve is reported.

**Deductive scorecard (new primary object).** q_0 = exact prior; K_l q(h) = E_{h'~q}[N(h; f_l(h'), sigma^2 I)];
q_{l+1} = (1 - delta) q~_{l+1} + delta W_{l+1}, with q~ from a family fitted to K_l q_l (moments / cells computed from q_l,
never from samples of p) and W a defensive wide component. Per step and family:
- coverage cost c_l = D_inf(K_l q_l || q_{l+1}) (bits): exact sup on a grid plus an analytic tail where the dimension
  allows (5-neuron case); elsewhere a lower estimate (sup over a large point set plus gradient ascent), labelled *estimated*;
- waste w_l = KL(q_{l+1} || K_l q_l) (bits), by quadrature over q (or MC over q where stated), with K_l q_l by
  quadrature over q_l;
- compute = kernel evaluations (and wall time);
- certified output statement p_L(A) <= 2^(sum c) q_L(A) for rare output events, compared with the true p_L(A)
  (validation sampling / quadrature) and with the sampling bound at equal compute (Clopper-Pearson 95% upper bound).
Families: Gaussian; product grid over named directions; activation-pattern mixture; k-sparse code; neuron code
(product over pre-activation groups z = W1 LN(h), groups = connected components of a weight-based gate-dependency graph).
Order of work: (E1) 5-neuron exact case, (E2) random vs trained small nets (waste growth per layer, Jacobian expansion
on q's support), (E3) tiny PFN layers at the analysis sigma, as far as tractable, with every estimated number labelled.

**E1 definition (my choice; the "5-neuron exact case" was not specified further in the brief).** x ~ N(0, I_2);
two noisy layers of 5 ReLU neurons each, s = LN0(ReLU(W h + b)) (centred, ||s||^2 = 5), h = s + sigma*eps; scalar readout
y = w.h_2. Rare events A_t = {y > t}. All densities by deterministic quadrature (Gauss-Hermite / grids).

**Predictions (A1).**
- P10. E1: with delta = 0.01 the best family's certified bound 2^eps q_L(A) is within 10x of the true p(A) for events
  with p(A) in [1e-6, 1e-3], and is tighter than the sampling bound at equal compute for p(A) <= 1e-4.
- P11. E1: activation-pattern mixtures have >= 2x the waste of the neuron-code family at layer 2.
- P12. E2: random nets in the ordered regime (gain <= 1.4) have per-layer waste <= 1 bit for the Gaussian family at
  sigma >= 0.3; waste grows with depth for gain 3 (chaotic); the trained net has more waste per layer than the random
  net of the same shape.
- P13. Task 2: the analysis sigma is <= 0.1 (SNR >= 100) for both PFNs.

## Amendment 2 (2026-10-28 ~18:25Z): fixed-size families with re-projection; width arm

Written after: E1 results for families G and XG2 (sigma 0.3: c_1 = 15.6 / 7.6 bits, waste 6.7 / 3.1 bits), the first
E2 layer-1 row of two nets (S = 1), PFN val gaps at sigma 0.1 (cost ~0.017 nats above sigma 0, both depths).
- **E2 (fixed size).** Families are mixtures of exactly S Gaussian cells (weighted k-means on the pushed node centres,
  cell moments + sigma^2 I) plus the delta defensive component; S never grows across layers; compute = L x cost(S).
  Per layer: KL waste delta_l(S) = KL(q_l || K q_{l-1}) and D_inf coverage c_l(S); S in {1, 4, 16, 64, 256};
  know-nothing cover (uniform on the LN0 sphere convolved with noise) as the upper reference; I(X;H_l) of the real net.
  Widths 8 / 16 / 32 at sigma 0.5 / 1.0 / 1.5 (C ~ 8.5 bits each); random g = 1, g = 3, trained.
- **Width arm.** 2L PFN at fixed SNR 30 and 10 (sigma 0.183, 0.316), widths 32 / 48 / 64 (wider nets warm-started by
  copying the base weights into the leading block; not function-preserving), same schedule; report KL to Bayes and
  agreement with the noiseless base.
Predictions:
- P14. For random g = 1 nets, delta_l(S) at fixed S varies by less than 2x across layers 2-6 (depth-independent);
  for g = 3 it grows with depth.
- P15. The S needed for KL waste <= 1 bit per layer grows with I(X;H_l); at equal capacity, width changes it little.
- P16. The know-nothing cover's KL waste per layer is >= 3x the S = 16 waste, and its D_inf coverage cost is the largest.
- P17. Width arm: at SNR 30 the width-64 net is within 0.01 nats of the noiseless base; at SNR 10 no width is.

## Amendment 3 (2026-10-28 20:35Z): Hölder per-event bounds with Rényi divergences

Written after all results in the report up to Update 7 (D_inf chains, decisive test, PFN diagnostics); before any
Rényi-divergence estimate exists.
- **Bound (one general model q):** for any event A at the level of p, q and alpha > 1,
  log2 p(A) <= (1 - 1/alpha) [log2 q(A) + D_alpha(p||q)]  (Hölder; D in bits). For a function 0 <= g <= 1 (an event pushed
  through the TRUE remaining kernels), log2 E_p g <= (1 - 1/alpha) [log2 E_q g^(alpha/(alpha-1)) + D_alpha(p||q)].
- **Composition (exact inequality):** weak triangle inequality (Mironov 2017, Prop. 11), for alpha > 1 and Hölder
  conjugates (p, q): D_a(P||R) <= ((a - 1/p)/(a - 1)) D_{p a}(P||Q) + D_{q (a - 1/p)}(Q||R); with data processing
  D_b(K P||K Q) <= D_b(P||Q) and P = p_l, Q = K_l q_{l-1}, R = q_l this gives
  B(l, a) = min_p [ ((a - 1/p)/(a - 1)) B(l-1, p a) + d_l(q (a - 1/p)) ], B(1, a) = d_1(a) (p_1 = K_1 q_0 exactly),
  d_l(b) = D_b(K_l q_{l-1} || q_l) (local, average-case), optimised over p per step (orders degrade backward).
- **Small nets:** the dec.py trained nets (width 8, depth 6, sigma 0.5 and 0.3), fixed-size chains S = 16 / 256 (+ delta
  defensive). d_l(b) on an order grid b in [1, 1024] by multiple importance sampling (pooled samples of K q and q), D_inf
  from the search. Direct check: D_alpha(p_l||q_l) with p_l's density from a 2^16-parent bank. Events per depth l:
  A = {w_l.h_l + b_l > t} with the least-squares task readout at layer l; truth by MC / subset simulation; ladder from
  p ~ 0.5 to 2^-40. Bounds: output-level (q_l(A), B(l, alpha)) and through-kernel (E_{q_{l-1}} g^(alpha'), B(l-1, alpha));
  also with the directly estimated D_alpha (composition-free). Sampling floor at equal wall time.
- **PFN (2L, sigma 0.05):** per-layer D_alpha in 512 dims is not estimable with node mixtures; reported instead:
  event-implied lower bounds D_alpha(p_l||q_l) >= (alpha/(alpha-1)) log2 p(A) - log2 P_l(A); the binned 1-dim output-level
  D_alpha(p_Y||q_Y) (a lower bound on any certificate's divergence) and the best-case Hölder bound it would give.
Predictions:
- P18. Small nets (S = 256, sigma 0.5): local d_l(2) <= 1 bit and d_l(8) <= 3 bits at every layer.
- P19. The composed B(L, alpha) is >= 50% of the D_inf-chain epsilon for alpha in {2, 4, 8} at L = 6 (composition
  re-introduces the worst case through order degradation), and >= 3x the directly estimated D_alpha(p_L||q_L).
- P20. alpha = 8 is the best of {2, 4, 8} for events at 2^-30 and below; alpha = 2 is best for p >= 2^-10.
- P21. Rare events (2^-20 .. 2^-40) at depth 6, sigma 0.5, S = 256: the best certified bound keeps <= 50% of the rarity
  bits (fails the "half the bits" target); at depth 1-2 it keeps >= 50%.
- P22. Common events (p in [0.05, 0.5]): q_L(A) is within a factor 1.5 of p(A), while every Hölder bound is >= 2x p(A)
  (the bound is loose for common events whenever D_alpha >= 1 bit).
- P23. The composed bound grows at least linearly with the number of composed layers (slope >= mean local d_l(alpha)).
- P24. PFN: event-implied D_alpha(p_r1||q_Gauss) >= 500 bits; best-case output-level Hölder bound with Bayes-mean cells at
  2^-41 loses <= 15 bits at alpha = 8.

## Amendment 4 (2026-10-28 22:55Z): surrogate chains (exact KL chain rule) and Donsker-Varadhan bounds

Written after Update 8 (Hölder/Rényi); before any surrogate-chain result.
- **Object.** Surrogate maps g_site run on the same path with the same Gaussian noise. Every noisy site has
  h_site | parents ~ N(f_site(parents), sigma^2 I), so for the path measures
  KL(real || surrogate) = sum_sites E_real ||f_site(parents) - g_site(parents)||^2 / (2 sigma^2) (nats, exact), estimated
  by plain sampling of real paths. Each term depends only on that site's surrogate, so the chain frontier is the sum of
  per-site frontiers.
- **Families** (fitted on 2^15 real paths, evaluated on fresh ones):
  - small nets f(h) = LN0(tanh(W h + b)): LR-r (rank-r pre-activation map, reduced-rank least squares, same tanh + LN0),
    LIN-r (rank-r affine map, no nonlinearity), TC-F-k (TopK transcoder with F features, k active, MSE-trained), exact;
  - PFN (2L): emb LR-r of W_in; attention sites: drop (cost 0), fixed average attention pattern (values exact), rank-r
    affine sublayer on the full 512-dim state; MLP hidden: LR-r of W1 per token, TC-F-k per token; MLP write: LR-r of W2.
  Description cost = parameters.
- **DV bounds** for output functions F (mean loss = squared error / NLL; output mean): for lambda > 0,
  E_p F <= (KL + log E_q e^{lambda F})/lambda and E_p F >= -(KL + log E_q e^{-lambda F})/lambda, lambda optimised on a
  grid; E_q by sampling the surrogate chain. Width compared with plain sampling's 95% CI at equal forward passes.
- **Skip-merge.** Two-site blocks with linearised propagated noise covariance sigma^2 (I + J J^T) (J = Jacobian of the
  second map at the noiseless intermediate input); Gaussian approximation checked against the sampled real two-step
  kernel; dynamic program over cut points minimising total KL + beta * parameters for several beta.
- **sigma trade-off.** Small nets at sigma 0.2 / 0.3 / 0.5; PFN 2L at sigma 0.05 / 0.1 / 0.183.
Predictions:
- P25. PFN sigma 0.05: every surrogate using <= 25% of a site's parameters has KL >= 100 nats at that site.
- P26. For a fixed family the per-site KL falls roughly like 1/sigma^2: from sigma 0.05 to 0.183 by 5-30x.
- P27. DV two-sided width on the mean loss is >= 10x plain sampling's 95% CI at equal compute for every PFN surrogate at
  sigma 0.05; on the small nets at sigma 0.5 some surrogate with <= half the parameters is within 3x of the CI.
- P28. Small nets: the linearised Gaussian two-step kernel is within 0.1 nats (KL) of the sampled real kernel at
  sigma 0.3 / 0.5; the DP chooses at least one merge at moderate beta, next to a strongly contracting layer.
- P29. Small nets: per-layer KL drops by >= 10x from the best rank-1 to the best rank-4 surrogate.

## Amendment 5 (2026-10-28 23:22Z): deviation prediction as the primary metric (Kelvin, 23:20Z)

Written after the surrogate-chain KL / DV results for the PFN at sigma 0.05 and 0.1 and the small nets (per-site KL,
DV widths), and after the review corrections were started; BEFORE any deviation-prediction or scenario number exists.
- **Deviation.** For a held-out episode x: network output = noise-marginalised predictive (K = 8 noise draws, mixture of
  the draws' mixture heads). Deviation = network minus exact Bayes for (i) predictive mean, (ii) predictive log-sd,
  (iii) per-episode KL(Bayes || network) (quadrature).
- **Prediction.** A surrogate chain (surrogate maps g at some sites, real maps elsewhere, same noise process) gives the
  same three quantities; its predicted deviation is (surrogate minus Bayes). Score: R^2 on 8,000 held-out episodes,
  R^2 = 1 - sum (d - d_hat)^2 / sum (d - mean d)^2.
- **Baselines.** "network = Bayes" (d_hat = 0); the untrained network of the same architecture at the same sigma;
  noise floor: R^2 between two independent K = 8 marginal estimates of the same network's deviation.
- **Per layer cut.** Single-site replacements (each site and family, others real) and cumulative replacements (all sites up
  to the cut). **Per seed:** 2L at sigma 0.05, seeds 4 (existing), 1 and 2 (fine-tuned now, same schedule); 3L seed 13.
  Precision bar: the surrogate's own error must be below the network's deviation (sd of d), i.e. R^2 > 0.
- **Secondary.** KL budget, description cost and DV bounds as before.
Predictions (deviation prediction):
- P30. Single-site replacements: near-full-rank surrogates (LR16 at emb / m / r) reach R^2(mean) >= 0.8; every surrogate
  with <= 25% of the site's parameters has R^2(mean) < 0.
- P31. Cumulative replacement of all MLP sites by LR16 (attention real): R^2(mean) >= 0.5; any attention surrogate
  (fixed pattern or rank <= 256) gives R^2(mean) < 0.
- P32. The untrained net has R^2 < 0 for all three quantities.
- P33. R^2 for the per-episode KL is below R^2 for the mean for every surrogate.
- P34. Noise floor R^2(mean) >= 0.95 at sigma 0.05.
**Scenario predictions** (fresh episodes; "ratio" = scenario average / all-episode average for the 2L sigma 0.05 seeds;
dmean = network minus Bayes predictive mean, dlogsd likewise; KL per episode):
- S1 bimodal Bayes posterior (top decile of between-component variance fraction): KL ratio >= 2 (predicted 2.5);
  network too unimodal / under-dispersed: mean dlogsd < 0 (predicted about -0.02).
- S2 strongly disagreeing query readings (top decile of reading spread) with ambiguous reliabilities (labelled-only
  27-way entropy above median): KL ratio >= 3 (predicted 3.5); mean dlogsd < 0.
- S3 all sources noisy (v = 2, 2, 2): RMS dmean ratio >= 1.3 (predicted 1.5); KL ratio >= 1.3.
- S4 all sources reliable (0.1, 0.1, 0.1): KL ratio <= 0.6 (predicted 0.45).
- S5 one reliable, two noisy (0.1, 2, 2 in any order): KL ratio >= 1.2 (predicted 1.4).
- S6 ID crosstalk (top decile of the largest |cosine| between different sources' ID vectors): RMS dmean ratio >= 1.1
  (predicted 1.25).
- S7 one labelled residual dominates a source (max r^2 / SS > 0.7): the tilted pooling (negative r^2 tilt) underestimates
  that source's noise, so the network over-trusts it: sign(dmean) = sign(o*_s - Bayes mean) in >= 60% of such episodes
  (predicted 65%).
- S8 extreme answers (|Bayes mean| > 2.5): shrinkage toward 0: mean dmean * sign(Bayes mean) < 0 (predicted -0.03),
  RMS dmean ratio >= 1.5.
- S9 strong query-reading coupling (top decile of KL(w_Bayes || w_lab)): KL ratio >= 2 (predicted 2.3).
A scenario is a hit if the predicted sign holds and the measured ratio / value is on the predicted side of the threshold.

## Amendment 6 (2026-10-28 23:57Z): lateral arm, sparse surrogate DAGs over token/source groups

Written after the seed-4 deviation-prediction results (single-site / cumulative chain surrogates, scenarios) and before
any DAG run. Phase A round 2 (work/cm-interp/ROUND2.md) found an own-source regime (2L two-pool seeds 1, 4; 3L after
block 1) and a cross-source regime (2L one-pool seeds 0, 2, 3 via a "mixer" head averaging the three query readings at
the QQ tokens; every 3L net after block 2).
- **DAG surrogate.** Nodes = (site, token); the surrogate of an attention site keeps the real weights but restricts each
  query token's keys to a sparse parent set (softmax renormalised over the allowed keys); token-local sites stay real.
  Independent per-unit noise makes the KL chain rule exact over the DAG: KL = sum_nodes E ||f - g||^2 / (2 sigma^2).
  Only ancestors of the output are scored (context tokens after a_L are not read).
- **Variants.** D1 per-source layer 1 (each source's 5 tokens attend only to their own source; QT attends to all); D1+lat
  (D1 plus QQ<->QQ lateral edges across sources); D2 (D1 with QT reading only the QQ tokens and itself in layer 1); D3
  (D2 with QT reading only QQ tokens and itself in layer 2); 3L: the same in layer 1, with layer 2 tested both
  per-source and with QQ<->QQ lateral edges.
- **Comparison at equal description cost.** Against the best chain surrogate for the attention sites from the earlier
  families (drop / fixed pattern / rank-r affine), on (a) KL budget and (b) deviation-prediction R^2 (mean, log-sd, KL)
  and the S1-S9 scenario hits. Description cost of a DAG variant = attention parameters + number of allowed
  (query, key) edges; treewidth of the moralised token-level DAG over all scored sites (networkx min-degree heuristic) and
  the largest clique's state dimension (tokens x 32) as the cost of exact (Gaussian) inference.
Predictions:
- P35. Own-source seeds (2L s1, s4): D1 costs <= 10% of the KL of the best chain attention surrogate and reaches
  R^2(mean) within 0.1 of the noise floor; one-pool seed 2: D1 R^2(mean) < 0, D1+lat recovers R^2(mean) >= noise floor - 0.1.
- P36. D2 and D3 (QT reading only QQ tokens) raise KL by < 2x over D1(+lat) and keep R^2(mean) >= 0.5 in the own-source seeds.
- P37. Treewidth of the token-level moral graph: exact chain >= 15; D1 / D2 <= 6 (+1 with lateral QQ edges).
- P38. Scenario hits of the best DAG equal the network's own hits on S1, S3, S6, S8 and S9 in every 2L seed.

## Amendment 7 (2026-10-29 00:42Z): question-weighted surrogate fits (coordinator, before the backward certificate)

Written after Update 9 (unweighted fits: compressed surrogates predict the deviation worse than "network = Bayes")
and before any weighted fit or K = 64 evaluation exists.
- **Question.** The network's deviation from Bayes per episode, features F = (mean, log-sd, KL(Bayes||net)) each divided
  by its across-episode sd (Update 9 values per model).
- **Weighted fit.** At site s, minimise E||J_s (g - f)||^2 + eps E||g - f||^2, J_s = per-sample Jacobian (3 x dim) of F
  (single noisy downstream pass, noise fixed) w.r.t. the site's output, by autograd on the real net; eps = 1e-3 x mean
  ||J||_F^2 / dim (keeps directions J does not see from drifting). Unweighted control: same parameterisation, eps-term only
  (output MSE), plus the Update 9 closed-form fits.
- **Families / compression levels.** emb, MLP hidden, MLP write: rank r in {2, 4, 8} (rank 2 and 4 are <= 25% of params
  except emb rank 4 at 26%); attention: fixed learnable pattern + rank-r value-output map, r in {2, 4} (18% / 36% of
  params), and fixed learnable pattern with full value-output (62%). Per-source DAG: restricted mask with all attention
  weights refitted by the weighted loss (D1; D1+lat; 3L: D1 and D1-L2lat).
- **End-to-end.** All sites at the <= 25% level (emb rank 4, attention pattern + rank 2, MLP rank 4), initialised from the
  weighted fits, then trained jointly to reproduce the network's predictive mean and log-sd per episode (common random
  numbers); also the rank-8 / pattern + rank 4 level.
- **Evaluation.** K = 64 noise draws per estimate (was 8), 2,000 held-out test episodes; noise floor = R^2 between two
  independent K = 64 estimates; R^2 of predicted vs actual deviation for mean, log-sd, KL; same models as Update 9
  (2L seeds 4, 1, 2; 3L seed 13).
**What each outcome would mean.**
- Weighted fits at <= 25% reach R^2(mean) > 0 where unweighted fits are < 0: the Update 9 headline was a property of the
  fitting loss, not of the network; compressed descriptions carry the deviation once told what matters.
- Weighted fits at <= 25% stay < 0: the deviation is spread over many directions per site even after weighting; the
  "fine-grained" reading of Update 9 stands.
- Weighted sites fine alone but the weighted chain fails while end-to-end succeeds: the sites' errors interact; descriptions
  must be fitted jointly. End-to-end worse than per-site weighted: an optimisation problem, not evidence about structure.
- Weighted DAG refit reaches the floor in own-source seeds: the earlier DAG gap was the renormalised real weights, not
  missing structure. Seed 2 still failing with lateral edges: the mixer needs other cross-source edges.
Predictions:
- P39. K = 64 noise floor: R^2 >= 0.95 (mean), >= 0.98 (log-sd, KL) in every model.
- P40. At rank 4, weighted single-site fits reach R^2(mean) >= 0.5 at every emb / MLP site where the unweighted fit is < 0;
  weighted attention (pattern + rank 2) reaches R^2(mean) > 0 but < 0.5.
- P41. The end-to-end <= 25% chain reaches R^2(mean) >= 0.5 and beats the chain of independently weighted fits.
- P42. Weighted D1 refit reaches R^2(mean) >= floor - 0.1 in the own-source seeds; seed 2 D1+lat refit reaches >= 0.5.


## Amendment 8 (2026-10-29 01:07Z): event-specific backward certificate (output martingale)

Written before any code for this test exists and before any result. Truth ladders are the existing ones
(`results/dec_d8_s{0.5,0.3}.json`, `runs/L2_s0.05/diag2.json`); nothing new about them is looked at first.
- **Construction (backward from A, event-specific).** Score Y (small nets: trained readout w.h_L + b; PFN 2L σ 0.05:
  predictive mean of one noisy pass), event A_t = {Y > t}. Backward functions φ_l(h) = E[Y | h_l = h] (the Doob
  martingale of the score; φ_L = Y, φ_0(x) = noise-averaged output for input/episode x). The increment at noise site l is
  a function of that site's noise Z_l with Lipschitz constant σ Λ_l, Λ_l = sup ||∇_{h_l} φ_l|| over reachable states, so
  it is conditionally sub-Gaussian (Gaussian concentration) and Azuma gives
  P(A_t) ≤ E_{x ~ exact prior}[min(1, exp(-(t - φ_0(x) - δ)_+^2 / (2 σ^2 Σ_l Λ_l^2)))] + P(escape).
  Reachable = site output on the sphere plus noise with ||Z_l|| ≤ ρ_l, ρ_l from the χ² tail at 2^-60 (escape mass and the
  shift δ it induces are added). Only the scalar φ_l enter: nothing about the rest of the 512-dim state is certified.
- **What is certified and what is searched.** Λ_l = max of ||∇φ_l|| over 8,192 real-path states (prior and tail-tilted
  inputs) and gradient ascent from 256 starts within the reachable set: a search sup, as for D_inf coverage before, not a
  formal verification. φ_0 by Monte Carlo over downstream noise (≥ 64 draws; its error is added as 3 standard errors);
  the expectation over the exact prior by importance sampling with a defensive mixture (small nets: Gaussian mean shift
  in x; PFN: query-label shift, as in Correction 3), with a 95% upper confidence bound. No sample of the rare event of
  the real net is used.
- **Diagnostics (not bounds).** (O1) the same formula with rms instead of sup gradient norms; (O2) P(φ_0(x) > t)
  (rarity carried by the input alone); (C2, PFN) the output ceiling: sup of the head's predictive mean over reachable
  final query states, by search, against the ladder's thresholds.
- **Report.** Certified rarity bits kept (-log2 bound / -log2 p) per rung from 2^-6 to 2^-40, Λ_l / rms_l per site, compute
  and plain sampling's 95% bound at equal compute.
**What each outcome would mean.**
- Bound within a few bits of the truth: an event-specific certificate needs only per-site worst-case sensitivities of
  one scalar; the full-state divergences of Updates 1-8 were the wrong object.
- Bound loose with Λ_l / rms_l large: the cost is "ψ right on unreachable states" (worst-case sensitivity), the program's
  backward-only failure; the next step is sensitivity bounds restricted by a forward description (the meeting).
- Bound loose with Λ_l / rms_l ≈ 1 (O1 also loose): the tail is not sub-Gaussian around φ_0 (e.g. saturation at a ceiling);
  the right backward object is the event's geometry, not a martingale bound.
Predictions:
- P43. Small nets (σ 0.5 and 0.3): sup/rms gradient-norm ratio ≥ 2 at ≥ 1 site; the certificate keeps between 40% and 80%
  of the rarity bits at 2^-40 and ≥ 70% at 2^-10.
- P44. Small nets: the certificate beats plain sampling's 95% bound at equal compute from 2^-30 on.
- P45. PFN: the head's searched sup over reachable final query states (C2) lies within 0.06 of the 2^-40 threshold (the
  ladder beyond 2^-14 is a ceiling effect of the head), and the certificate keeps ≥ 70% of the rarity bits at 2^-10 but
  ≤ 50% at 2^-40.
- P46. O1 (rms) is within 3 bits of the truth down to 2^-20 on both small nets; on the PFN it over-predicts by > 3 bits
  beyond 2^-16 (near the ceiling the increments are smaller than any typical-state sensitivity).


## Amendment 9 (2026-10-29 01:43Z): E2, whole-network structure built from the question (coordinator, 01:3xZ)

Order change (coordinator): E1 (amendment 7) → E2 (this) → E3 → E4 → the backward certificate of amendment 8 (its
pre-registration stands unchanged; it runs last). Each claim gets a Codex review before it enters the report.
Written before any E2 code or result; E1 single-site rows for seed 4 (emb only) and the E1 chain/DAG rows were seen.
- **Models.** 2L seeds 4, 1, 2 and 3L seed 13 at σ 0.05 (as E1).
- **Nodes and weights.** Node = (site, token). Node weight w(s,t) = E_x ||J_{s,t}||², J = Jacobian of the deviation
  features F (E1's normalised mean, log-sd, KL) w.r.t. that node's output, real net, one noisy downstream pass, 4,096
  episodes from the prior. Edge (layer l, query t, key t') weight e = E_x ||J_{a_l,t} · S_t · m_{t'→t}||², m = the head-summed
  attention message from t' to t, S_t = Jacobian of the sphere projection (first-order effect of deleting the message).
- **Stage A, backward build (real maps, structure only).** From the output node (query token at the last residual) go
  backward: each needed token at layer l keeps the smallest key set covering (1 − ε) of its edge weight; its keys become
  needed at layer l − 1 (plus itself through the residual). Tokens never needed are not ancestors of the output. ε in
  {0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0}. Restricted attention renormalises the real weights.
- **Stage B, local moves (greedy, 12 rounds)** from the stage-A description with the largest ε whose proxy R²(mean) ≥
  0.95 × that of ε = 0. Candidates each round: prune one edge group (layer × query group × key group; groups = source
  0 / 1 / 2 context tokens, query token); drop one node group to its fit-set mean (site × {source s labelled tokens,
  source s query-question token, query token}); pool one sibling group (lateral merge: the three query-question tokens,
  or one source's labelled tokens, share their mean state at a site); named-pattern attention at layer l; MLP hidden or
  write map at rank 8 or 4; embedding at rank 8; skip-merge of MLP block l at rank 8 (isotropic noise inflated to the
  propagated variance). Attention named pattern: per head, logits = θ · φ(t, t'), φ = 11 named features (same source,
  same question, key is query-question token, key labelled, query row × key query-question, query row × key labelled,
  self, key-source reliability r_s = −log(mean squared labelled residual of source s + 0.01), r × query row, r × same
  source, query row × query key): 22 parameters per layer instead of 2,112 for q and k; real value/output maps.
  Surrogate fits: output-MSE initialisation, then question-weighted loss with anchor ε_anchor = 0.1 × mean||J||_F² / dim
  (E1 showed 1e-3 drifts), then end-to-end tuning of that site inside the real net on the single-draw predictive mean and
  log-sd with common random numbers (300 steps). Score = Δ proxy R²(mean) per Δcost; take the best move if it saves
  cost; moves with Δ R² ≥ 0 first.
- **Cost.** Parameters = unique floats the description uses (real maps counted once per layer, surrogate parameters,
  named-pattern θ, dropped-node constants) + kept attention edges + kept nodes (structure, one unit each). KL bits =
  Σ over scored nodes E||f − g||² / (2σ² ln 2) on fit episodes. Both reported; the greedy uses parameters + structure.
- **Proxy score (search only).** R² between the description's and the network's deviation with common random numbers
  (K = 8, 1,024 fit-distribution episodes). **Reported score** on every Pareto point: E1's K = 64 independent-draw R²
  on the 2,000 test episodes, and scenario agreement (S1-S9 outcomes vs the network's, 32,768 fresh episodes, K = 8).
- **Rediscovery.** Fraction of kept layer-1 context→context edges that are within-source, per ε, versus the hand-built
  per-source DAG (D1); nobody tells the procedure about sources.
Predictions:
- P47. Seeds 4 and 1: at the stage-B start ε, ≥ 80% of kept layer-1 context→context edges are within-source; seed 2 < 60%.
- P48. Named-pattern attention costs ≤ 0.2 proxy R²(mean) at layer 1 in seeds 4 and 1, and > 0.3 at the last layer.
- P49. The best whole-network description at ≤ 50% of the network's parameters has K = 64 R²(mean) ≥ 0.5 in seeds 4 and
  1, and < 0.5 in seed 2 and the 3L net.
- P50. Pareto points with K = 64 R²(mean) ≥ 0.5 reproduce ≥ 7 of the 9 scenario outcomes.


## Amendment 10 (2026-10-29 02:01Z): E3, descriptions relative to Bayes

Written before any E3 code has run (no smoke test either) and before any E2 result.
- **Free inputs.** b(x) = 24 exact Bayes quantities of the episode, all functions of the observed readings and labels:
  posterior mean and log-sd of y*; per-source reliability posteriors from the labelled data (3 × 3); per-source
  reliability posteriors including the query-reading coupling (3 × 3); each source's posterior weight in the mean (3);
  bimodality (between-component / total variance). z-scored on fit episodes. Cost counts only parameters beyond them.
- **E3a, whole network relative to Bayes.** Predictive mixture head (4 components) = MLP_h(b), h in {0 (linear), 8, 16,
  32, 64} (two hidden layers of width h). Trained on 16,384 fit episodes to minimise E_x E_z KL(network single-draw
  predictive || description), fresh noise every step (equivalent to fitting the noise-marginal predictive). Scored as E1
  (K = 64, 2,000 test episodes, R² of mean / log-sd / KL deviation) and by scenario agreement (S1-S9 outcomes vs the
  network's, 32,768 fresh episodes).
- **E3b, every E2 surrogate gets b.** E2's procedure rerun from the same stage-A start with these candidate changes: low-rank
  maps and the skip-merge take b into their bottleneck (+ 24 × r parameters); named attention uses the exact reliability
  (labelled-posterior expected log precision of the key's source) instead of the residual statistic, plus the source's
  posterior weight (query row); one new candidate replaces the whole last block for the query token by
  sphere(x_q + MLP_16(b)) (944 parameters), which removes every last-layer attention edge.
- **Comparison.** Pareto curves (cost vs K = 64 R²(mean) and scenario agreement) of E3a, E3b and E2 on the same models.
**What each outcome would mean.**
- E3a at a few hundred parameters near the floor: the network's deviation is a low-dimensional function of the exact
  computation; mechanism-level descriptions are only needed for what Bayes quantities do not carry.
- E3a far below the floor while E2 does better: the deviation depends on things Bayes does not compute (e.g. identity
  crosstalk between source embeddings, S6); a description relative to Bayes must add those nuisances.
- E3b's Pareto curve dominating E2's: the Bayes quantities are what the network's internal states encode; the description
  cost of the network is mostly the cost of computing Bayes.
Predictions:
- P51. E3a with h ≥ 16 reaches K = 64 R²(mean) ≥ 0.6 in every model but stays ≥ 0.1 below the floor.
- P52. E3a misses the ID-crosstalk scenario (S6) in every model where the network shows it, and reproduces ≥ 6 of 9 outcomes.
- P53. At equal cost, E3b's best R²(mean) exceeds E2's by ≥ 0.1 at every cost below 50% of the network's parameters.


## Amendment 11 (2026-10-29 02:45Z): E4, payoff in compute

Written before any E3 or E4 result exists (E2 is running; its stage-A rows were seen).
- **Description.** The E3a head (predictive mixture = MLP_h(b), exact Bayes quantities b as free inputs) with the width h
  that has the best K = 64 test R²(mean) for that model in E3a, refitted inside E4 with the E3a recipe so its setup cost is
  measured. If no E3a width reaches R²(mean) ≥ 0.3, E4 still runs and reports that the description is poor.
  E2/E3b descriptions keep real attention and MLP maps, so one evaluation costs about one network pass; they cannot pay
  off in compute and are not used.
- **Questions (fixed now).** Q1–Q10: average KL(Bayes ‖ network's noise-marginal predictive, K = 64) over the episodes of
  each scenario S1–S9 and over all episodes; scenario thresholds (quantile-based S1, S2, S6, S9) fixed from a 16,384-episode
  reference sample. Q11–Q12: P(|deviation of the predictive mean| > τ), τ at the ground truth's 99% and 99.9% quantiles.
- **Ground truth.** The network on 32,768 fresh episodes, K = 64 each; S8 also from 8,192 episodes with the query label
  tilted to ±3 (exact weights).
- **Estimators at budget B** (forward passes; B = 2^14 … 2^21): plain sampling of the network with B / 64 episodes at
  K = 64 (relative RMSE over 8 repetitions, drawn from an independent 32,768-episode pool); the description, available
  once B exceeds its setup cost (network passes used to fit it plus its own training time at the measured pass rate), with
  its evaluation time also converted to passes. Its error = relative error of its answers on 32,768 fresh episodes.
- **Models.** 2L seeds 4 and 2.
Predictions:
- P54. The description's relative error is ≤ 25% on the KL questions for S3, S4, S5, S8 and overall, and > 25% for S6
  (identity crosstalk is not a Bayes quantity).
- P55. Its setup costs ≥ 10^6 network-pass equivalents; at B = 2^21 plain sampling's relative RMSE is ≤ 10% on the
  overall and common-scenario KL questions, so the description beats sampling at equal compute on at most 3 of the 12
  questions (candidates: S8, the 99.9% tail, S2).


## Amendment 12 (2026-10-29 04:44Z): E5, systematic vs idiosyncratic deviation (coordinator, before the certificate)

Written before any E5 code or result. The amendment-8 certificate runs had already finished when this was added; their
results are not looked at before E5 is reported.
- **Systematic part.** S(x) = E[deviation | z(x)] for the mean, log-sd and KL deviations (K = 64), with z(x) = named
  episode features: the scenario variables of S1–S9 (bimodality, reading spread, labelled-posterior entropy, query
  coupling, max source-ID |cosine|, max residual share per source, |Bayes mean|, the true reliability levels), the 24 exact
  Bayes quantities of E3, the labelled-posterior reliability of each source, disagreement (sd and range of the query
  readings), extremeness (|Bayes mean|, max |query reading|), ID overlap (the three pairwise source-ID cosines).
  Estimated by held-out regressors fitted on 65,536 fresh episodes (K = 64 targets): gradient-boosted trees with a small
  grid chosen on a validation split, and a 2-hidden-layer MLP; the better of the two on validation is used.
- **Systematic fraction** per seed and feature = held-out R² of S(x) against the network's K = 64 deviation on the 2,000
  test episodes, divided by the noise floor.
- **Learnability ceiling.** A fresh 2-layer transformer of the PFN's shape (no noise, random init) trained on the same 65,536
  episodes to predict the three deviation features from the raw tokens; held-out R² on the 2,000 test episodes.
- **Scoring compact descriptions on the systematic part only:** R² of each description's predicted deviation against S(x)
  on the 2,000 test episodes, for (a) the per-source DAG (D1, real weights; D1 + lateral for seed 2), (b) the lowest-cost
  E2 path points (the 72% point with all four MLP maps at rank 8, and the path's cheapest point; no path reached 50%),
  (c) the E3a Bayes head of the best width, (d) "scenario mechanisms": a linear regression of the deviation on the
  scenario variables, fitted on the same 65,536 episodes. Also their R² against the full deviation, as before.
- **Models.** 2L seeds 4, 1, 2 and the 3L net.
**What each outcome would mean.**
- Small systematic fraction for the mean: most of the per-episode mean deviation is idiosyncratic function-approximation
  error; a compact explanation can only be expected to predict the systematic part, and should be scored there.
- Ceiling well above the systematic fraction: the remainder is learnable from the input but not expressible in the named
  features, i.e. structured but not in these terms.
- Compact descriptions scoring high on S(x) but low on the full deviation: they explain the systematic part.
Predictions:
- P56. Systematic fraction of the mean deviation ≤ 0.3 in every model; of log-sd and KL ≥ 0.5.
- P57. The learnability ceiling for the mean is between the systematic fraction and 0.6 of the floor.
- P58. On S(x): the scenario-mechanism regression and the E3a head reach R² ≥ 0.5 for log-sd and KL; the per-source DAG
  reaches ≥ 0.7 on all three features in seeds 4 and 1; the 72% E2 point reaches ≥ 0.8 on all three in every model.
- Addendum to amendment 12 (before any full E5 run; after a smoke test at toy sizes): a raw description's deviation includes
  idiosyncratic structure that S(x) averages out, so R² against S(x) also penalises correct idiosyncratic predictions. A
  second, post-hoc-labelled score is added: R² of S(x) against E[description's deviation | z] (5-fold cross-fitted trees on
  the test episodes), with the network itself (independent draws) scored the same way as the reference.


## Amendment 13 (2026-10-29 05:28Z): E6, shared vs seed-specific deviation (coordinator)

Written before any E6 code, training or result.
- **Nets.** The three existing 2L σ = 0.05 nets (bases mix4_ws_s4, _s1, _s2) plus five new ones trained with the same
  noise recipe (`nnoise.py train --sigma 0.05 --steps 20000 --cooldown 8000 --ramp 1000`): two more of the same lineage
  recipe (bases mix4_ws_s0, mix4_ws_s3; training seeds 5, 3) and three from the from-scratch mixture recipe (bases mix4_s0,
  mix4_s1, mix4_s4; seeds 6, 7, 8; mix4_s3 excluded, its base failed with gap 0.13). Main set = the five same-recipe nets
  (10 pairs). Cross-recipe pairs are reported separately, excluding pairs whose bases share a seed index (they may share
  an initialisation).
- **Deviations.** Mean, log-sd and KL deviation from Bayes on the first 8,000 test episodes, K = 64, two independent
  replicates per net (as E1).
- **Shared fraction.** For nets i ≠ j: ρ_ij = Cov(d_i, d_j) / sqrt(V_i V_j), V_i = Cov(d_i rep 1, d_i rep 2) = the
  noise-free variance of net i's deviation. Under d_i = s + u_i + noise with seed-specific parts u_i independent across nets,
  ρ_ij = Var(s) / Var(s + u), the shared share of the noise-free deviation: an upper bound on what any explanation of how
  this architecture and task err can capture, if seed-specific parts are independent. Reported per feature as the mean
  over pairs, with a 95% bootstrap CI over episodes (1,000 resamples) and the range over pairs.
- **Comparison with E5.** E5's named-feature R² put on the same scale: R² / (V_i / Var(d_i)), for seeds 4, 1, 2.
- **Step 2, run only if** the shared fraction of the mean deviation exceeds the named-feature fraction by ≥ 0.1 at the
  lower end of its CI: regress the seed-averaged deviation (65,536 fit episodes, K = 64, all nets of the main set) on (a)
  E5's named features, (b) those plus token-level structure (all source-ID and question-ID cosines, all 15 readings and 4
  labels, labelled residuals); held-out R² against the seed-averaged test deviation, as a fraction of its shared variance.
  Score the per-source DAG and the replayed E2 72% point of seeds 4 and 1 against the seed-averaged test deviation.
- **Step 3, run only if** the shared fraction of the log-sd deviation is ≥ 0.4 at the lower end of its CI: (a) the three
  named quantities with the largest permutation importance for the seed-averaged log-sd deviation; (b) a compact additive
  model (a piecewise-linear function of each of the three, 8 knots) fitted on fit episodes, scored by held-out R² against
  the seed-averaged and each net's log-sd deviation; (c) localisation in seed 4: ridge probes from each site's state
  (query token, and per source for the layer-1 sites) to the net's own log-sd deviation and to the three quantities, to
  find where the systematic spread error first becomes linearly readable.
Predictions:
- P59. Shared fraction (main set): mean ≤ 0.25; log-sd ≥ 0.5; KL between 0.2 and 0.5.
- P60. For log-sd, E5's named-feature fraction is within 0.15 of the shared fraction; for the mean it is lower by ≥ 0.05.
- P61. Cross-recipe shared fractions are within 0.1 of the same-recipe ones.
- P62 (if step 3 runs). The compact 3-quantity model reaches ≥ 0.6 of the seed-averaged log-sd deviation's shared
  variance, and the per-net log-sd deviation becomes linearly readable (≥ half its systematic part) only at the last
  residual of the query token.
- Addendum to amendment 13 (after the first stats run on the main set, before any step 2/3 result): the first run paired
  replicate 1 of both nets, which shares the noise generator seeds across nets; the cross-net covariance now pairs
  replicate 1 of one net with replicate 2 of the other (both orders averaged), so no noise draw is shared. The first run's
  values (mean 0.48, log-sd 0.60, KL 0.48) are superseded by the rerun.


## Amendment 14 (2026-10-29 06:43Z): E7, intervention on the spread error, grouping on fresh nets (coordinator)

Written before any E7 code, training or result.
**E7.1 Does the readout create the pull toward a typical width?** Nets: the eight E6 nets. Evaluation: the first 4,000
test episodes, noise-marginal predictive with K = 64 draws (as E6); the log-sd deviation is the target, mean and KL are
reported too. Interventions, per net:
- I1 derived readout: per noise draw, predictive N(μ̂, exp(ℓ̂)²) with μ̂, ℓ̂ = ridge decodes of the signed Bayes mean and the
  Bayes log-sd from the a2 query state, fitted on 16,384 fresh fit episodes of that net (never on evaluation episodes);
  I1c the same with decoders calibrated by reverse regression (no least-squares attenuation).
- I2 retrained head: net frozen through r2 (noise on); a new head of the same form (LayerNorm + linear → 4-component
  mixture) trained from scratch with the training objective (single-pass NLL of y*) for 6,000 steps on fresh episodes.
- I3 retrained readout from a2: m2, r2 and the head replaced by a 2-layer MLP (width 64) from the a2 query state to a
  4-component mixture, trained as I2.
- I4 ablations: (a) MLP2's write replaced by its mean over fit episodes; (b) MLP2 skipped (r2 = a2).
- I5 noise off at every site (one deterministic pass). I6 noise off at the layer-2 sites (a2, m2, r2) only.
- I7 (diagnostic) single-draw predictive: the average over the 64 draws of each draw's log-sd deviation.
- Decomposition of the baseline: marginal log-sd deviation = A + B, A = 0.5 log E_z[var_single] − Bayes log-sd
  (per-pass spread), B = 0.5 log(1 + Var_z(mean_single) / E_z[var_single]) (between-draw jitter of the predictive mean).
Metric: shared spread error retained R = Cov(d_int, d̄₋ᵢ) / Cov(d_orig, d̄₋ᵢ), d̄₋ᵢ = average of the other seven nets'
original log-sd deviations (replicate 2); removed = 1 − R. Also the pull slope (OLS slope of the log-sd deviation on the
Bayes log-sd) and the mean offset. Reported per net and averaged over the eight, with bootstrap CIs over episodes.
**What each outcome would mean.**
- I5/I6 or I7 remove ≥ 60%: the pull comes from marginalising over the injected noise (between-draw jitter added to a
  spread that each pass already calibrates for), not from the readout.
- I2/I3 remove ≥ 50%: the learned readout (head or MLP2) creates the pull.
- I2/I3 keep ≥ 70% while noise-off removes it: the pull is what the training objective prefers given a noisy
  representation (information-limited), not a defect.
- I1/I1c remove it while I2/I3 do not: the encoded quantities suffice, but NLL-trained readouts do not use them that way.
- I4 removes it without destroying the mean: MLP2 creates it.
Predictions:
- P63. The jitter term B carries ≥ 50% of the shared spread error (covariance share) on average; I5 removes ≥ 60%; I7
  removes ≥ 40%.
- P64. I2 and I3 keep ≥ 70% of the shared spread error.
- P65. I1 (least-squares decoders) keeps ≥ 50%; I1c removes ≥ 50%.
**E7.2 Does the post-hoc 4-vs-4 grouping hold on fresh nets?** New nets with the E6 noise recipe: from the unused
from-scratch base mix4_s2 and from three new from-scratch bases (tiny-pfn `pfn.py train --mixk 4`, seeds 5, 6, 7, 90,000
steps). Bases or nets with a validation gap above 0.03 nats are excluded and reported. For each new net the per-source DAG
test (D1, real weights, K = 64, first 2,000 test episodes) assigns a predicted group before any cross-net correlation is
computed: A if DAG R²(mean) > 0.3, else B. The predictions are written with a timestamp to
`results/e7_group_predictions.json`. Test (E6 estimator, 8,000 test episodes): an A-predicted net is confirmed if its mean
mean-deviation correlation with the four A nets is ≥ 0.6 and exceeds that with the four B nets by ≥ 0.1; a B-predicted
net is confirmed if its mean correlation with the A nets is ≤ 0.55.
- P66. The assignment is confirmed for ≥ 3 of the 4 new nets.
**E7.3 (if time).** The five-net seed-averaged mean deviation regressed on token-level features plus interactions (query
reading × each source-ID cosine, query reading × labelled residual sums, products of query readings, per-position
reading/label/residual), as in E6 step 2.
- P67. Interactions raise the finite-ensemble score for the mean from 0.49 to at most 0.6.


## Amendment 15 (2026-10-29 09:15Z): E8, source mis-weighting and layer-2 attention patches (coordinator)

Written before any E8 outcome. Seen beforehand (design aid, `qattn_look.py`, 2,048 fit episodes): in every group-A net at
least one layer-2 head puts ~0.8 of the query row's attention on the three query-question (QQ) tokens, with per-source
mass correlating 0.9–0.97 with the Bayes source weights; group-B nets put more mass on labelled tokens and have a head
whose source mass correlates only −0.05 to 0.4 with the Bayes weights.
Nets: the 12 nets of E6/E7. Group A = seed 4, seed 1, mx0, mx1, mx2, nb6; group B = seed 2, ws0, ws3, mx4, nb5, nb7.
**E8.1 Source mis-weighting.**
- Implied source weights ŵ_s(x) = ∂ m(x) / ∂ o*_s: gradient of the network's noise-marginal predictive mean (K = 8 draws)
  with respect to source s's query reading (its token's value feature), by autograd. Bayes weights W_s(x) = ∂μ_B/∂o*_s by
  central finite differences of the exact Bayes mean. Weight error δ_s = ŵ_s − W_s. Fit set: 8,192 fresh episodes;
  test set: the first 4,000 test episodes.
- Linearised mean deviation from weight errors: Σ_s δ_s (o*_s − c), with c = 0 or c = μ_B, whichever has the higher R² against
  the net's own mean deviation on fit episodes (chosen once, on the all-net average). With the measured δ this is the
  ceiling for any mis-weighting description.
- Compact model: one additive piecewise-linear function g (6 knots per input), shared by the three sources, of six named
  per-source quantities: log of the source's labelled squared-residual sum, log of the other two sources' sums (smaller,
  larger), the source's Bayes weight, |o*_s − mean of the other two readings|, and the source's max |ID cosine| with the
  others. Fitted by least squares to the group-averaged δ_s on fit episodes (all nets, group A, group B separately).
- Score: R² of Σ_s g(·)(o*_s − c) against the group's seed-averaged test mean deviation (8,000 test episodes where available,
  replicate 1), on the finite-ensemble scale of E6 (R² × Var(average) ÷ mean within-group cross-net covariance).
- **What counts as "explains the shared mean error":** compact score ≥ 0.5 in a group = explains; 0.25–0.5 = partly;
  < 0.25 = does not. For reference: E7.3's black box reached 0.65 (all nets).
**E8.2 Patches of the layer-2 query row** (each head; all other computation real; K = 64, first 4,000 test episodes):
- Pbayes: each source's total attention mass (its five tokens) set proportional to its Bayes weight W_s, keeping the head's
  total context mass and the within-source split.
- PbayesQQ: only the three QQ-token masses set proportional to W_s (their total kept); labelled tokens unchanged.
- Ptrue: source masses proportional to the true precisions 1/v_s (oracle reliability).
- Puniform: equal source masses (control: no source weighting).
- Pmean: the query row replaced by its average over fit episodes (control: no episode dependence).
- Measures: retained covariance (E7 definition) of the log-sd deviation and of the mean deviation with (a) the
  leave-one-out average of the other nets of the same group and (b) of all other nets; also the change in mean squared
  deviation from Bayes. Reported per net and averaged by group.
**What each outcome would mean.**
- Pbayes removes ≥ 50% of a group's shared mean error (retained ≤ 0.5) while Puniform and Pmean do not: the shared mean
  error is mis-weighting at layer-2 attention.
- Pbayes also removes ≥ 50% of the shared spread error: the spread error comes from the same source weighting (location
  found). If not: the spread error is set elsewhere.
- Ptrue doing better than Pbayes: the net's error partly follows the Bayes-vs-truth gap (it is "too right").
- Patches that remove the shared pattern but raise the mean squared deviation: the layer carries the error but the patch
  breaks other computation.
**E8.3 (conditional).** If in group A the compact model scores ≥ 0.5, or Pbayes removes ≥ 50% of the shared mean or spread
error, the mechanism is stated in a few sentences and tested on the held-out fresh A net nb6: the compact model and
patch are refitted or rerun without nb6 in any average, and the test is R² of the compact model's predicted mean deviation
against nb6's own deviation, compared with the R² of the other five A nets' average deviation against nb6. Holds if
the compact model reaches ≥ 0.5 of that reference (and, for a patch, if Pbayes on nb6 retains ≤ 0.5 of nb6's shared error).
Predictions:
- P68. With the measured δ, the linearised deviation reaches R² ≥ 0.6 against the all-net seed-averaged mean deviation.
- P69. The compact model scores ≥ 0.5 in group A and < 0.5 in group B.
- P70. In group A, Pbayes retains ≤ 0.5 of the shared mean error and ≤ 0.7 of the shared spread error; Puniform and Pmean
  increase the mean squared deviation by ≥ 50%.
- P71 (if E8.3 runs). The held-out test holds on nb6.


### Amendment 15, addendum (2026-10-29 09:22Z), before any E8 patch outcome or score
- Found while writing the analysis: the launched patch code set the "Bayes" source masses from the Bayes posterior
  precision share (qbayes features 20:23, sum over configurations c of p(c | data) times source s's share of the
  precision under c), not from W_s = dmu_B/do*_s as written above. On the test episodes the two correlate 0.95 after
  normalisation (mean |difference| 0.04; in 5% of episodes some source differs by more than 0.22), and W_s is negative
  for 1.8% of source entries. Both patch runs were stopped before writing any output and restarted with
  Pbayes and PbayesQQ = W_s clipped at 0 and renormalised to sum 1 (as written; clipping is needed for attention masses),
  plus two extra variants on the same noise draws, Ppost and PpostQQ = the posterior precision share.
- The implied-weight pass (all 12 nets) finished before this addendum. Seen from it: only the per-net log line
  "mean |w_hat - W| on fit episodes" (0.030-0.032 for the six A nets, 0.040-0.047 for the six B nets).
- Analysis details (no threshold changes): the compact model's "Bayes weight" input is W_s; compact-model scores use all
  8,000 test episodes (W_s for test episodes 4,001-8,000 computed the same way); the measured-delta ceiling uses the
  first 4,000 (where w_hat exists); c is chosen on the all-net average fit-episode deviation (K = 8 predictive mean from
  the implied-weight pass); knots = pooled deciles 0.1-0.9 (6 points) of each input on fit episodes; patch measures use the
  E7 definition (other nets' replicate-2 deviations; denominator = the same net's base variant on the same noise
  draws); group averages get 500-sample episode-bootstrap 95% intervals.


### Amendment 15, addendum 2 (2026-10-29 09:29Z): E8.1b, post hoc, written after the pre-registered E8.1 failed
- Seen before writing this: the pre-registered linearisation fails. On fit episodes (all-net average) R² is −6.5 with c = 0
  and −1.6 with c = mu_B (c = mu_B chosen); the measured-delta ceiling on test episodes is R² −1.8 (all nets), −1.9 (A),
  −1.3 (B); the compact scores are negative in every group. These pre-registered results stand: P68 fails and E8.1, as
  pre-registered, "does not explain".
- Diagnosis (also seen): the implied weights are reliable (two independent noise seeds: per-source correlation of delta
  0.96; noise sd 0.009 against signal sd 0.046), and the linearised prediction correlates 0.37 with the all-net average
  deviation but has 1.8 times its sd. The local-derivative linearisation also fails for noise-free reference predictors that
  differ from Bayes only in how they treat the sources (labelled-only predictor: correlation 0.92, R² −0.56; question-only:
  R² 0.44), because a local derivative includes the shift of the configuration posterior and does not integrate linearly.
  So the pre-registered measure cannot answer whether the mean error is mis-weighting.
- E8.1b (post hoc, run once as follows; same nets, episodes and score scale). Exact path decomposition
  dev(o) = dev(o_ref) + sum_s (o*_s − c) dbar_s, with o_ref = (c, c, c), c = mu_B(o), and dbar_s = the average of
  (w_hat_s − W_s) along the straight path from o_ref to o (8 Gauss-Legendre nodes; the same K = 8 noise draws at every
  node; identity check on 300 test episodes of seed 4: R² 1.0000). Measures per group (all, A, B), on the finite-ensemble
  scale, with the pre-registered thresholds (≥ 0.5 explains, 0.25-0.5 partly, < 0.25 does not):
  (a) agreement offset: the group-average dev(o_ref) against the group's seed-averaged test mean deviation (first 4,000);
  (b) reading-combination part (ceiling for any mis-weighting description): the group-average sum_s (o*_s − c) dbar_s;
  (c) compact model on dbar: the same g (six inputs, 6 knots, shared over sources) fitted to the group-average dbar_s on
      fit episodes, scored as sum_s g (o*_s − c) on the 8,000 test episodes;
  (d) compact model fitted directly: the same form fitted by least squares to the group-average fit-episode mean deviation
      (K = 8 predictive mean), scored the same way;
  (e) held-out nb6 for (c) and (d), as in E8.3 (refit on the other five A nets; holds if its R² on nb6 is ≥ 0.5 of the
      R² of the other five A nets' average).
- Predictions (post hoc): P72. (b) scores ≥ 0.6 and (a) ≤ 0.2 for all nets. P73. (c) or (d) scores ≥ 0.5 in group A.
  P74. (e) holds on nb6 for the better of (c) and (d).
- E8.2 and the E8.3 trigger stay as pre-registered.


### Amendment 15, addendum 3 (2026-10-29 09:31Z): E8.2b, post hoc, written after seeing only patch MSD ratios
- Seen before writing this: only the per-net mean-squared-deviation ratios (mean deviation) of the E8.2 patches for three
  group-A nets (seed 4, seed 1, mx2), printed by the run: Pbayes 5.4-9.0x, PbayesQQ 5.0-8.5x, Ppost 2.5-4.4x,
  PpostQQ 1.5-3.5x, Ptrue 9.0-14.3x, Puniform 6.7-9.5x, Pmean 10.0-13.3x. No retained-covariance value seen.
  Every full patch breaks the computation, so the full-patch retained covariance mixes removal of the shared pattern with
  new error.
- Added (E8.2b, run once on all 12 nets, K = 64, first 4,000 test episodes, the same noise draws as E8.2): a dose-response
  patch, per-source masses m'_s = (1 − a) m_s + a M Wt_s (M = the head's total source mass; Wt = W_s clipped and
  renormalised; within-source split kept), a = 0.1, 0.25, 0.5 on both heads (PbA10, PbA25, PbA50); and the full and
  half patch on one head only, the head with the largest average query-row mass on the QQ tokens on fit episodes (PbH,
  PbH50).
- Measures: E8.2's retained covariance (same-group leave-one-out, replicate 2) and MSD ratio, by group. Read-out: a cause
  share phi = (1 − retained) / a from the a = 0.1 and 0.25 points (the share of the shared error that moves with the layer-2
  source masses, to first order). phi ≥ 0.5 with the MSD falling at a = 0.1 = layer-2 source mis-weighting carries most of
  the shared error; phi ≤ 0.25 or the MSD rising at a = 0.1 = it does not.
- Prediction (post hoc): P75. In group A, phi ≥ 0.5 for the mean error and the MSD falls at a = 0.1.


### Amendment 15, addendum 4 (2026-10-29 10:06Z): E8.4, post hoc, written after seeing E8.1, E8.1b and E8.2 results
- Seen: E8.2's retained covariances (group A: Bayes-type patches keep 0.24-0.47 of the shared mean error, Puniform 1.38,
  Pmean 1.60; spread error kept by every patch in A) and partial E8.2b (5 of 12 nets). E8.1b: the reading-combination
  part carries 0.93 of the shared mean error but the six-quantity compact models score at most 0.19.
- E8.4 tests the mechanism the patches point to, with no flexible fit: the mean error is the net's layer-2 value
  weighting of the query readings minus the Bayes value weighting. Net weights q_s = the query row's attention mass on
  source s's QQ token, summed over heads and normalised over the three QQ tokens (average over K = 4 noise draws, layer-1
  noise included). Bayes value weights p_s = the posterior precision share (qbayes features 20:23) normalised to sum 1,
  so that mu_B = (sum of the unnormalised shares) x sum_s p_s o*_s exactly. Model M1: dev = k x sum_s (q_s − p_s) o*_s, one
  parameter k; control M0 (flat weighting): dev = k x sum_s (1/3 − p_s) o*_s. For a group, the term is averaged over its
  nets; k is fitted by least squares to the group-average fit-episode mean deviation (K = 8, from the implied-weight pass);
  scores on the 8,000 test episodes on the finite-ensemble scale, with the registered thresholds. Secondary (reported, not
  used for verdicts): source masses (all five tokens) instead of QQ masses, and clipped W_s instead of p_s.
- Held-out nb6: k fitted on the other five A nets; prediction = k x nb6's own term; holds if its R² against nb6's own mean
  deviation is ≥ 0.5 of the R² of the other five A nets' average deviation (0.55).
- Descriptive: per group, the slope of q_s on p_s (pooled over sources and episodes) and the mean entropy of q and p.
- Predictions: P76. M1 scores ≥ 0.5 in group A, and more than M0. P77. M1 holds on nb6. P78. In group A the net's QQ weights
  are flatter than p (slope < 1).


## Amendment 16 (2026-10-29 10:27Z): E9, where the layer-2 attention masses become the net's effective source weights (coordinator)

Written before any E9 outcome. Seen beforehand:
- Design aid from E8 data (first 4,000 test episodes): every net's effective source weights, normalised, have slope
  0.97-1.02 on the Bayes posterior share p, while its layer-2 QQ masses have slope 0.48-0.77. The net turns flat masses
  into Bayes-sharp weights somewhere after the masses.
- Smoke test (seed 4, 100 episodes): the rebuilt forward equals the model's (difference 0); the six path parts sum to
  the full gradient (1e-7). Seed 4's fit diagnostics: the reading explains 0.29-0.31 of each QQ value vector's variance;
  a linear map explains 0.91 of MLP2's query-row write.
Nets: the 12 nets of E8 (A = seed 4, seed 1, mx0, mx1, mx2, nb6; B = seed 2, ws0, ws3, mx4, nb5, nb7). Evaluation: first
4,000 test episodes; "disagreeing" = the third of them with the largest sd of the three query readings.
**E9.1 Exact path split** (`qe9.py`). Effective weights w_s = d m / d o*_s (noise-marginal predictive mean, K = 8 draws,
autograd). The layer-2 query-row computation has three inputs: R = the query token's own residual x1_q; M = the attention
logits (queries and keys); V = the values. Each is split into its direct route to the output and its route through
MLP2. The six parts sum to w exactly. Sharpening = Delta_s = w_s − (sum_t w_t) q_s, where q = the query row's QQ masses
(summed over heads, normalised over sources, same draws). Each part's share of the sharpening = sum(Delta^P Delta) /
sum(Delta Delta), with Delta^P defined in the same way from the part's weights (shares sum to 1). Reported per net and
averaged by group, by path (R, M, V), by route (direct, MLP2) and for all six parts.
- Reading: share ≥ 0.5 = that part carries the transformation; 0.25-0.5 = contributes; < 0.25 = does not.
**E9.2 Patches** (K = 64 deviations with E8's noise draws; K = 8 effective weights). Parameters are fitted on the 8,192 E8 fit
episodes (one noise draw), never on test episodes:
- PVg: each QQ token's layer-2 value (per head) keeps everything except its coordinate along the fitted reading
  direction b (OLS of the value on the reading o*_s), which is set to the reading itself. This removes episode-dependent
  gain on the reading.
- PV: each QQ token's value replaced by its fitted linear function of the reading, a + b o*_s (removes all other content).
- PM: MLP2's query-row write replaced by its least-squares linear approximation in the post-attention state (same noise
  draw on the MLP hidden layer).
- PVgPM and PVPM: both together.
Masses are never changed.
Measures:
- Sharpness S = slope of the normalised effective weights on p (episodes with sum of w > 0.2). S_q = the same for q.
  Flattening f = (S_base − S_patch) / (S_base − S_q).
  - Reading: f ≥ 0.5 = the patched component does at least half of the transformation; 0.25-0.5 = part of it;
    < 0.25 = not.
- Retained covariance (E7/E8 definition, same-group leave-one-out replicate-2 reference) for the mean and log-sd
  deviations, on all 4,000 episodes and on the disagreeing third.
- MSD ratios (mean, log-sd) and the mean KL ratio, on both sets.
**What each outcome would mean.**
- PVg f ≥ 0.5 and PM f < 0.25: the values carry reliability-dependent gain on the reading. The transformation is in the
  value path (made by layer 1's encoding of the QQ tokens), not in MLP2.
- PM f ≥ 0.5 and PVg < 0.25: MLP2's nonlinearity sharpens the weights.
- Only the combinations reach 0.5: both are needed.
- PV ≥ 0.5 but PVg < 0.25: the transformation uses the values' non-reading content, not gain along the reading.
- No patch ≥ 0.5: neither value gain nor MLP2's nonlinearity does it. E9.1's M and R shares then say whether readings
  move the masses or reach the output through layer 1's residual.
- Shared error, for any patch:
  - retained ≥ 1.2 with MSD ≥ 1.5 while f ≥ 0.5: removing the sharpening adds to the shared error, so that error is
    partly under-sharpening;
  - retained ≤ 0.5 with MSD ≤ 1.5: the component carries the shared error itself without breaking the net;
  - retained ≤ 0.5 with MSD > 1.5: it moves the pattern but breaks the net;
  - retained 0.8-1.2: the component does not touch it.
  - The same reading applies to the spread error.
Predictions:
- P79. In group A, the value path (both routes) carries ≥ 0.5 of the sharpening.
- P80. In group A, the MLP2 routes together carry < 0.25 of it.
- P81. In group A, PVg has f ≥ 0.5 and PM has f < 0.25.
- P82. Every patch with f ≥ 0.5 raises the shared mean-error covariance (retained ≥ 1.2) and the mean MSD (≥ 1.5).
- P83. For every patch with f ≥ 0.25, the mean-error retained covariance is further from 1 on the disagreeing third than on
  all episodes.


### Amendment 16, addendum 1 (2026-10-29 10:36Z): E9.3, post hoc, written after seeing E9.1/E9.2 for three A nets
- Seen: E9.1 and E9.2 for seed 4, seed 1 and mx2 (all group A). The value path carries 0.96 of the sharpening (direct
  route 0.50, through MLP2 0.46); R −0.07; M 0.11. PVg flattens the weights (f 0.84-1.00) but multiplies the mean MSD by
  about 32. PM: f 0.07-0.38, mean MSD x5.9.
- Because PVg breaks the net, a non-destructive check is added. Reading gain of each QQ value: g[h, s] = derivative of
  source s's layer-2 value coordinate along its fitted reading direction (head h) with respect to its own reading
  (K = 8, same draws as E9.1, through layer 1). Per-source gain = the mass-weighted average over heads.
  Masses x gains: u_s = sum_h A[h, s] g[h, s], normalised over sources.
- Measures per net, averaged by group:
  - corr(per-source gain, p_s);
  - sharpness of u (slope on p);
  - share of the sharpening it reproduces, (S_u − S_q) / (S_base − S_q);
  - corr(normalised u, normalised effective weights).
- Reading: share ≥ 0.5 = the values' reading gains account for at least half of the sharpening (descriptively);
  0.25-0.5 = part; < 0.25 = no.
- Prediction (post hoc): P84. In group A, the share is ≥ 0.5 and corr(gain, p) ≥ 0.5.


### Amendment 16, addendum 2 (2026-10-29 10:40Z): E9.4, post hoc, written after seeing the E9.3 preview
- Seen: the E9.3 preview for seed 4, seed 1 and mx2. The gain along the single fitted reading direction is negative in
  11-20% of source-episodes, and masses x gains correlate only 0.02-0.14 with the effective weights (masses alone 0.85-0.92).
  So a gain along one fixed direction is not a usable per-episode quantity, and E9.3 cannot test the value-gain account.
- E9.4 asks instead which values carry the value path's sharpening, with no direction fitted. For each source s, a pass
  with only s's own QQ-token value attached (masses and residual detached; both routes; E9.1's draws) gives the own-value
  part of w_s. The cross part = the value path minus own (other QQ tokens, labelled tokens and the query token, through
  layer-1 mixing). Shares of the sharpening as in E9.1; check: all tokens attached = V_direct + V_mlp (1e-7 on 200
  episodes).
- Reading: own share ≥ 0.5 = the sharpening is carried by how each source's own QQ value responds to its reading, as read
  out downstream (built by layer 1). Cross share ≥ 0.5 = it is routed through other tokens' values.
- Prediction (post hoc): P85. In group A, the own share is ≥ 0.5.


## Amendment 17 (2026-10-29 11:05Z): E10, each source's response against the Bayes requirement, episode by episode (coordinator)

Written before any E10 outcome. No new E10 quantity has been computed. Known from earlier E8/E9 runs, which E10 reuses:
- path-averaged weight errors (qig);
- local weights (qweights);
- masses (qattnw);
- the E9 path shares.

**What "the Bayes requirement" means here.**
- Hold the net's other routes (masses, mass path, residual, other tokens) as they are. For the net to match Bayes, source s's
  own-value response would have to change by exactly the weight error. So the departure of the value response from what
  Bayes needs is the weight error itself:
  - locally, delta_s = w_s − W_s;
  - along the path from agreement (E8.1b), dbar_s, which gives the reading-combination part of the mean deviation exactly.
- In gain units (per unit attention mass), the departure is delta_s / q_s. E9 put about 0.94 (A) of the sharpening on the
  value route, which is why the departure is expressed as a value-gain correction. That framing does not claim the value
  route alone causes the departure.
- Data: the 12 E8 nets; the first 4,000 test episodes (where dbar, w and q exist) for description; 8,192 fit episodes and
  8,000 test episodes for prediction.

**E10.1 Where it departs (descriptive).**
- Each source in each episode gets a reliability rank (1 = largest posterior share p_s) and a disagreement
  D_s = |o*_s − mean of the other two readings|, in terciles.
- Per group, report the group-average dbar_s (and delta_s, and dbar_s / q_s) in each rank x D-tercile cell, with the
  range over nets and the number of nets that share its sign.
- Report the same for the "disagreeing third" of episodes (largest sd of the three readings).

**E10.2 Does the departure predict the shared mean error on held-out episodes?**
- A flexible model of the departure: HistGradientBoosting (learning rate 0.05, 31 leaves, min 40 per leaf, up to 500
  iterations, 10% early-stopping split, seed 0). Target: the group-average dbar_s on the 8,192 fit episodes (rows =
  episode x source). 16 source-symmetric features:
  - own: p_s, clipped normalised W_s, log own labelled SS, o*_s − mu_B, D_s, max |ID cosine|;
  - the two other sources, ordered by p: p, log SS, o* − mu_B;
  - episode: sd of the readings, mu_B, Bayes log-sd, entropy of p.
  - No network internals.
- Prediction on the 8,000 held-out test episodes: sum_s (o*_s − mu_B) dbar_hat_s.
- Score as in E8: R² against the group's seed-averaged test mean deviation, times Var(average), divided by the mean
  within-group cross-net covariance; 500-sample bootstrap. Also on the disagreeing third.
- Secondary: the same structured form, sum_s (o*_s − mu_B) g(features_s), with g a small MLP (2 x 64, GELU, Adam, early
  stopping on 10%) fitted directly to the group-average fit-episode mean deviation (K = 8).
- Reference (not held out): the measured test dbar gives 0.92 (A) and 0.98 (B) (E8.1b).
- Reading: score ≥ 0.5 = the departure predicts the shared mean error well; 0.25-0.5 = partly; < 0.25 = no. Per group
  (A, B) and all nets.

**E10.3 (only for a group where E10.2's primary score is ≥ 0.5): does layer 1 build the reliability signal the values
use?**
- For each source's QQ token, a ridge decoder of standardised log own labelled SS (the reliability evidence; lower = more
  reliable) from its layer-1 output x1 (32 dims). Fitted on fit episodes with one noise draw. Shift direction u_s =
  beta / |beta|², so x1 + Delta u_s moves the decoded value by Delta SD.
- Patch: the shift, applied only where layer 2 computes the values. Keys, queries and the residual keep the original x1.
  Delta = ±0.25 SD for all three sources at once; central differences. Controls:
  - (i) the same shift applied to keys and queries only;
  - (ii) a random direction orthogonal to beta with the same norm, applied to the values.
- Net response: d w_s / d Delta (autograd effective weights, K = 8). Bayes response: d W_s / d Delta, with source s's
  labelled residuals scaled by exp(Delta x sd / 2), so that log SS moves by Delta SD.
- Accuracy: mean-deviation MSD ratio at Delta = ±0.25 (K = 64, E8 draws).
- Supported if all four hold:
  - decoding R² ≥ 0.5 on test;
  - the pooled slope of the net's response on Bayes's response is ≥ 0.5, with correlation ≥ 0.5, for the value shift;
  - |slope| ≤ 0.1 for the random control;
  - MSD ratio ≤ 1.1 at both signs.
- Reported per net and by group.
Predictions:
- P86. In both groups, in the top D tercile, the most reliable source is under-weighted (dbar < 0) in at least 5 of 6 nets.
- P87. E10.2's primary model scores ≥ 0.5 in group A.
- P88. If E10.3 runs, it supports the layer-1 hypothesis in group A.


### Amendment 17, addendum 1 (2026-10-29 11:07Z): E10.2b, post hoc, written after seeing E10.1 and E10.2
- Seen:
  - E10.1: the most reliable source is under-weighted (dbar < 0) in all 6 nets of both groups in every D tercile, and the
    least reliable source is over-weighted in all 6.
  - E10.2: the primary model scores 0.35 (A), 0.39 (B) and 0.42 (all), "partly". Its dbar prediction R² on test is
    0.41-0.46; the direct MLP scores 0.19 (A) and 0.36 (B).
  - So E10.3's registered trigger is not met.
- E10.2b asks whether the shortfall comes from missing input features: E10.2's HGB, fitted the same way, with 13 added
  input features. The inputs give each episode random source and question identity vectors, and the net's processing
  may depend on their geometry. Added per source row:
  - own labelled residual magnitudes, sorted (4);
  - ID cosine with each other source, ordered by p (2);
  - own ID-vector norm (1);
  - cosines of the query question's ID with the four labelled questions' IDs, sorted (4);
  - raw readings, own and the other two ordered by p (keeps 2 new: own o*_s and o* of other1; other2 follows from mu_B).
- Reading as registered (≥ 0.5 = predicts well).
- If E10.2b reaches ≥ 0.5 in a group, E10.3 runs for that group as registered, reported as triggered by a post-hoc model.


### Amendment 17, addendum 2 (2026-10-29 11:10Z): E10.3 triggered; one-at-a-time shifts and E10.2b feature groups (post hoc)
- E10.2b scores 0.60 (A) [0.55, 0.64] and 0.69 (B) on held-out test episodes, so E10.3 runs for both groups as
  registered, reported as triggered by a post-hoc model.
- Added before running E10.3. The registered shift moves all three sources' decoded reliability together. That mostly
  tests a common change, under which Bayes rescales all the weights together. So a one-at-a-time version is added
  (secondary, post hoc): the value shift (and the random control) applied to one source's QQ token at a time. Responses
  are split into own (the shifted source's weight) and cross (the other two weights). Bayes is computed the same way,
  with only that source's labelled residuals scaled. Same Delta, draws and readout.
- E10.2b feature groups (informative, post hoc): refit E10.2b without one group at a time and report the change in
  score. Groups: the E10.2 base features; labelled residual magnitudes; source-ID geometry (cosines and norm); query-
  question ID cosines; raw readings.


### Amendment 17, addendum 3 (2026-10-29 11:11Z): descriptive follow-up (post hoc)
- Seen: in the E10.2b feature groups, dropping the labelled residual magnitudes costs the most among the added groups
  (A: 0.60 to 0.41; B: 0.69 to 0.44). Bayes uses a source's labelled residuals only through their sum of squares (SS).
- Follow-up: the concentration kappa_s = largest squared labelled residual / SS (0.25 = spread evenly, 1 = all in one
  question). Report:
  - the partial correlation of dbar_s with kappa_s, given log SS_s and p_s (linear), on the first 4,000 test episodes;
  - this per net and for the group average, with the number of nets of each sign;
  - mean dbar_s by kappa tercile within log-SS terciles.
- Descriptive only; no threshold.


### Amendment 17, addendum 4 (2026-10-29 11:16Z): E10.4, descriptive, post hoc, written after seeing E10.3 on three A nets
- Seen: E10.3 for seed 4, seed 1 and mx2.
  - Decoding R² is 0.93.
  - Shifting the decoded reliability in the values barely moves the weights: slope 0.02 on the Bayes response; one at a
    time, own slope −0.00.
  - The shift keeps accuracy (MSD ratio 1.01-1.02).
- E10.4 asks whether the reliability-dependent response is a size effect rather than a direction: the reading's
  sensitivity being scaled, for example by the spherical normalisations in layer 1. Per source and episode (first 4,000
  test episodes, K = 8, E9 draws):
  - |d x1_QQs / d o*_s| (the layer-1 output's sensitivity to its own reading);
  - |d v_QQs,h / d o*_s| averaged over heads (the value's sensitivity);
  - the downstream-read own gain G_s = w_own_s / q_s (E9.4's own-route weight per unit of QQ mass).
- For each: correlation with p_s and with log own SS, per net and group (sign counts).
- Descriptive only; no threshold.


### Amendment 17, addendum 5 (2026-10-29 11:30Z): E10.5, post hoc, written after seeing E10.3 and E10.4 on all 12 nets
- Seen:
  - E10.3: not supported in any net. The decoding R² is 0.93 (A) and 0.82 (B), but the value-route slope on the Bayes
    response is 0.03 and 0.01, with accuracy kept (MSD ratio 1.00-1.04).
  - E10.4: the size of the reading's sensitivity in the QQ layer-1 output correlates with p (0.74 in A, 0.56 in B; 12 of
    12 nets positive) and with log own SS (−0.90 and −0.67). The downstream-read own gain correlates 0.81 (A) and 0.69 (B)
    with p.
  - So layer 1 may set the reliability dependence as the size of the reading's encoding. A constant shift of x1 (E10.3)
    cannot change that size.
- E10.5 tests this with small doses that keep accuracy. Only where layer 2 computes the values, the reading-driven part
  of each QQ token's layer-1 output (x1 − x1ref, with x1ref = layer 1 at readings set to mu_B) is scaled by
  lambda_s = (gbar / g_s)^a. Here g_s is E10.4's sensitivity size and gbar the episode's geometric mean over sources.
  a = 0.1 and 0.25 equalise that fraction of the log-size differences between sources. Control: the same scalings with
  each episode's source sizes permuted at random. K = 8 effective weights (E9 draws); K = 64 deviations (E8 draws).
- Measures (group means, per-net ranges): flattening f(a) (as E9.2) and f(a)/a; MSD and KL ratios; E8's retained
  covariance with the shared mean-error pattern.
- Reading: the size of layer 1's reading encoding carries the sharpening if all hold:
  - f(a)/a ≥ 0.5 at both doses;
  - the control's |f/a| ≤ 0.1;
  - mean MSD ratio ≤ 1.1 at a = 0.1.
- Prediction (post hoc): P89. This holds in group A.


## Amendment 18 (2026-10-29 12:04Z): E11, fresh-episode confirmation and matched error-spread pairs (coordinator)

Written before any E11 computation. No fresh episode, pair or E11 model exists yet.

**E11.1 Fresh-episode confirmation of the post-hoc departure model (E10.2b).**
- Frozen form: exactly E10.2b (`qdep2.py`). The 16 `qdep.features` plus 13 `qdep2.extra` features; HistGradientBoosting
  (learning rate 0.05, 31 leaves, min 40 per leaf, max 500 iterations, 10% early stopping, seed 0).
- Targets: the group-average path-averaged weight error dbar on the 8,192 E8 fit episodes, for the groups all, A and B.
- The three models (and, as a reference, the registered 16-feature E10.2 models) are fitted and pickled before the fresh
  episodes are generated. The pickle's SHA256 goes into prereg_hashes.txt.
- Fresh episodes: 8,000 new episodes, from W.sample_np(8000, np.random.default_rng(20_261_029)) and W.tokens_np with the same
  generator. Bayes quantities from W.references; W_s by central differences.
- Each of the 12 nets' mean deviation on them: K = 64, two replicates with independent noise (seeds 1 and 7, as E6).
- Score as in E10.2: R² against the group's seed-averaged deviation (replicate 1) × Var ÷ the mean within-group cross-net
  covariance (r1/r2 pairing); 500-sample bootstrap; also on the disagreeing third.
- Confirmed in a group if the frozen E10.2b model scores ≥ 0.5 there. Also reported: the change from its earlier test
  scores (0.68 all, 0.60 A, 0.69 B) and the frozen registered E10.2 model.

**E11.2 Matched error-spread pairs.**
- Base: 1,500 new episodes (np.random.default_rng(20_261_030); IDs as in W.tokens_np, held fixed within each set).
- For each base episode and each source s, four variants differ only in s's four labelled errors r_q = o_s,q − y_q. All
  keep s's actual sum of squares SS_s. The largest squared error is kappa x SS_s and the other three are (1 − kappa)/3 x
  SS_s each, for kappa = 0.25 (all equal), 0.45, 0.65 and 0.85. The signs and the position of the largest error are drawn
  once per episode and source and shared across kappa. Everything else is identical: labels, query readings, other
  sources and identity vectors.
- Check before any net is run: the Bayes mean, variance and W_s agree across the four variants to 1e-8; otherwise stop.
- Per net:
  - the modified source's effective weight w_s (autograd, K = 8, E9 draw scheme);
  - the predictive mean and log-sd (K = 64, the same noise draws for all variants of an episode).
- Primary contrast: kappa = 0.85 against 0.25. r = (w_s(0.85) − w_s(0.25)) / W_s, where W_s is Bayes's weight, identical
  within the set. Average over base episodes and sources; group mean over nets; 500-sample bootstrap over base episodes.
- **The effect counts** in a group if all three hold:
  - |mean r| ≥ 0.05: one concentrated error changes the weight by at least 5% of Bayes's weight against the same total
    spread evenly;
  - the same sign in ≥ 5 of 6 nets;
  - the group's 95% interval excludes 0.
- Also reported:
  - the same contrast in the top SS tercile (unreliable sources) and by reliability rank;
  - the shift of the predictive mean toward the source's reading;
  - the change in predicted log-sd;
  - the full kappa curve.
**E11.3 (only for a group where E11.2's effect counts).** What the nets compute instead, as an observation.
- A one-parameter family of reliability statistics T_p = c_p (sum_q |r_q|^p)^(2/p). c_p makes E[T_p] = E[SS] for four
  standard-normal errors; p = 2 is Bayes's SS.
- For each variant: the "pseudo-Bayes" weight, i.e. Bayes's weight with s's errors replaced by evenly spread ones whose
  sum of squares is T_p.
- Choose p on the grid 0.25-2.0 (step 0.25) to minimise the squared difference between the group-average observed weight
  differences (each kappa level against 0.25) and the pseudo-Bayes ones. Report p and the fit R², and say whether the
  statistic resembles a known robust estimator. Observation only.
Predictions:
- P90. The frozen E10.2b model scores ≥ 0.5 on the fresh episodes in both groups.
- P91. E11.2's effect counts in both groups with r > 0, and is larger in the top SS tercile and in group B.
- P92. If E11.3 runs: the best p ≤ 1.5 in group A.


### Amendment 18, addendum 1 (2026-10-29 12:35Z): robustness of E11.2's ratio measure (post hoc)
- Seen: E11.2 on all 12 nets.
  - The registered r counts in B (+0.103 [0.055, 0.142], 6/6) but not in A (+0.018, 4/6).
  - The Bayes weights W_s of the modified source include near-zero and negative values (min −0.53; 1% quantile −0.016;
    10% quantile 0.047), so per-row ratios can be extreme. The median r is −0.005 (A) and +0.029 (B); the ratio of means is
    +0.002 and +0.062.
- Added robustness checks (reported beside the registered verdict, which stands):
  - ratio of means sum(dw) / sum(W) overall and by SS tercile;
  - mean r over rows with W_s ≥ 0.05;
  - per net, the sign of sum(dw) in the top SS tercile;
  - the output-level shift of the predictive mean toward the source's reading as a ratio of sums, by SS tercile.
  - Bootstrap intervals over base episodes for the group values.


### Amendment 18, addendum 2 (2026-10-29 12:35Z): E11.3 by SS tercile (post hoc)
- Seen: the robustness checks.
  - In A the effect changes sign with SS: top tercile +0.13 ratio of means (6/6 nets), middle −0.04 (0/6), bottom −0.006.
  - In B it is positive and grows with SS: +0.003, +0.12, +0.22.
  - The registered E11.3 fit over all rows gives p = 2.0 (A; R² 0) and 1.75 (B; R² 0.06).
- Added: the same one-parameter T_p fit separately within each SS tercile (grid 0.25-4.0, now including p > 2, which makes
  the largest error count more), per group. Observation only.


## Amendment 19 (2026-10-29 12:41Z): E12, confirmation of the large-error finding on fresh matched pairs (coordinator)

Written before any E12 data is generated. The claim, ranges, measures, bars and code path below are frozen.
**Frozen claim.** For sources with large labelled errors, the big-miss version of a matched pair gets more weight than the
spread version, in all 12 nets.
- Big miss: kappa = 0.85, i.e. the largest squared labelled error is 85% of the source's sum of squares SS, the other three
  equal.
- Spread: kappa = 0.25, all four squared errors equal.
- Same SS, signs and position; exact Bayes identical.
**Frozen ranges, from E11's 4,500 source rows** (SS = the source's sum of squared labelled errors over its four labelled
questions):
- large: SS ≥ 3.182394 (E11's 2/3 quantile);
- middle: 0.679992 ≤ SS < 3.182394 (between E11's 1/3 and 2/3 quantiles).
**Fresh pairs.**
- Generated after this amendment's hash: 3,000 new base episodes from W.sample_np(3000, np.random.default_rng(20_261_031)).
- Then from the same generator, in E11's order: source and question identity vectors (as W.tokens_np); random signs (n, 3, 4);
  random big-error positions (n, 3).
- Every (episode, source) row whose SS falls in the large or the middle range gets the two variants (kappa 0.85 and 0.25),
  built exactly as in `e11_data.py` (largest error² = kappa·SS, the other three (1 − kappa)·SS/3 each, shared signs and
  position). Nothing else changes.
- Check before any net is run: the Bayes mean, variance and weights agree across the two variants to 1e-8; otherwise stop.
**Per net (all 12).**
- The modified source's effective weight w (qe9.eff_weights, K = 8, E9 draw scheme).
- The predictive mean (qsys.net_dev, K = 64, seed 1; the same noise draws for both variants by position).
- Δw = w(0.85) − w(0.25).
**Primary test (large range).**
- Per net: the mean of Δw over large-range rows.
- Pooled: the mean over the 12 nets, with a 95% interval from 1,000 bootstrap resamples of base episodes (each resampled
  episode brings all its rows).
- **Confirmed if** the per-net mean Δw is > 0 in at least 10 of 12 nets **and** the pooled interval excludes zero (lies
  above it).
- Effect size: Σ Δw / Σ W per net and pooled (E11 post hoc: +0.13 in A, +0.22 in B), where W is Bayes's weight.
**Secondary test (group A's reversal at middle range).**
- Per A net: the mean Δw over middle-range rows.
- **Replicates if** it is < 0 in at least 5 of 6 A nets **and** the A-pooled 95% interval (same bootstrap) lies below zero.
- E11 post hoc: ratio of means −0.041, 0 of 6 nets positive.
**Also reported (no bar):**
- group B in the middle range;
- the shift of the predictive mean toward the source's reading, as Σ (Δm × sign(o* − mu_B)) / Σ (W |o* − mu_B|), in both ranges;
- per-net values.
Predictions:
- P93. The large-range claim is confirmed (≥ 10 of 12 nets positive; pooled interval above zero).
- P94. Group A's middle-range reversal replicates (≥ 5 of 6 negative; A interval below zero).


## Amendment 20 (2026-10-29 12:58Z): E13, does layer 1's encoding strength carry the error-spread effect? (coordinator)

Written before any E13 quantity is computed. Data: the E12 fresh matched pairs (`results/e12_pairs.npz`; 2,962 large-range
and 3,006 middle-range rows; big miss = kappa 0.85, spread = kappa 0.25). E12's weight results are known; the encoding
strengths and patches below have not been computed.
**E13.1 Does the encoding strength respond to error spread like the weights do?**
- Encoding strength of the modified source s: g = |d x1_QQs / d o*_s|, the norm of the layer-1 output's sensitivity to
  that source's own reading (as E10.4).
  - Computed by forward-mode differentiation, with the Jacobian averaged over K = 8 noise draws (seeds 2 + 1000 x batch,
    batch 500) and then normed.
  - Computed for both versions of every pair. Δg = g(big miss) − g(spread).
- Per net: the mean Δg in each range, and Σ Δg / Σ g(spread). Pooled: the mean over nets, with a 95% interval from 1,000
  bootstrap resamples of all 3,000 E12 base episodes (empty clusters kept).
- **Matches the weight pattern** if all three hold:
  - (i) large range: Δg > 0 in ≥ 10 of 12 nets, and the pooled interval lies above 0;
  - (ii) middle range, group A: Δg < 0 in ≥ 5 of 6, and the A interval lies below 0;
  - (iii) middle range, group B: Δg > 0 in ≥ 5 of 6, and the B interval lies above 0.
- Also reported (no bar): the per-net correlation across pairs between Δg and E12's Δw.
**E13.2 Causal check that keeps accuracy.**
- In the big-miss version, only for the modified source and only where layer 2 computes the values, the reading-driven
  part of its QQ token's layer-1 output is scaled by lambda = g(spread) / g(big miss). This is E10.5's construction:
  x1' = x1ref + lambda (x1 − x1ref), with x1ref = layer 1 at readings set to mu_B and the same draw. It sets the big-miss
  version's encoding strength to the spread version's.
- Measured: the patched big-miss effective weight w' of the modified source (K = 8, E9 draws). Removed fraction:
  R = Σ (w_big − w') / Σ (w_big − w_spread), pooled over the 12 nets and the rows of a range, with the same bootstrap; also
  per net.
- Control: the same lambda applied instead to one of the other two sources (chosen at random per row, seed 13).
- Accuracy: K = 64 mean deviations of the big-miss episodes, patched against unpatched, on the same noise draws (seed 1).
  MSD ratio = mean patched² / mean unpatched².
- **The encoding strength carries the effect (through the value route)** if all four hold:
  - pooled R ≥ 0.5 in the large range;
  - R > 0 in ≥ 10 of 12 nets;
  - the control's |R| ≤ 0.1;
  - the mean MSD ratio over nets ≤ 1.1.
- Reading: R 0.25-0.5 = partly; < 0.25 = no. Group A's middle range is reported the same way (removal of a negative
  difference), with no bar.
**What each outcome would mean.**
- E13.1 matches and E13.2 carries: the error-spread departure is computed in layer 1 and expressed as how strongly each
  source's reading is encoded. The same mechanism that sets reliability-dependent weights also carries this departure
  from Bayes. The one-sentence statement is then updated to say so.
- E13.1 matches but E13.2 removes < 25%: the encoding strength moves with the departure, but the weight difference
  reaches the output mainly by other routes (masses, keys, residual).
- E13.1 does not match (no change, or a different sign pattern): the departure does not arise in the encoding strength. It
  enters elsewhere, for example in layer-2 masses.
- Anything in between: partial.
Predictions:
- P95. E13.1 matches the weight pattern (all three conditions).
- P96. E13.2: the encoding strength carries the effect (all four conditions).


## Amendment 21 (2026-10-29 13:23Z): E14, which function of a source's labelled errors sets layer 1's encoding strength? (coordinator)

Written before any E14 quantity is computed. Known beforehand:
- E10.4's encoding strengths g on the first 4,000 test episodes;
- E13's g on the E12 pairs (big miss stronger at large errors in all nets; weaker at middle errors in A, stronger in B).
The idea that a count above a threshold could produce A's reversal was reasoned out from E12/E13, not fitted.
**E14.1 Which form fits?**
- Target: g = |d x1_QQs / d o*_s| per (episode, source) (as E10.4: forward mode, Jacobian averaged over K = 8 draws, seeds
  2 + 1000 x batch, batch 500).
- Fit set: E10.4's values (`runs/NET/qsens.npz`, the first 4,000 test episodes x 3 sources).
- Held-out set: computed now, the same way, on the 8,000 E11 fresh episodes (`results/e11_fresh.npz`) x 3 sources.
- Inputs: the source's own four labelled errors r_q = o_s,q − y_q.
- Forms (all symmetric in the four errors):
  - F1 SS = sum r² (the Bayes statistic);
  - F2 SA = sum |r|;
  - F3 MAX = max |r|.
  - F1-F3 are mapped to g by a piecewise-linear function of log(statistic), with knots at the fit-set deciles (least squares).
  - F4 COUNT_tau = number of |r_q| > tau, mapped by per-count means. tau is chosen per net on the fit set from 0.1, 0.2, ..., 3.0.
  - F5 learned symmetric function: an MLP (2 x 32, GELU) on the sorted log|r| (four inputs). Fitted on the fit set with Adam,
    early stopping on 10%, seed 0. This is the ceiling for functions of the four errors.
  - Informative ceiling, not residual-only: HGB on the sorted |r| plus o*_s − mu_B, the other two sources' log SS, max
    |ID cosine| and the sd of the readings.
- Score: held-out R² of g per net; group mean and range.
- **A named form fits well** in a group if both hold:
  - its group-mean held-out R² is ≥ 0.9 x F5's;
  - F5's is ≥ 0.5.
**E14.2 Does it explain A's reversal? Is it the same in A and B?**
- Each fitted form (per net) is applied to the E12 pairs' error vectors, giving a predicted Δg per row. This is compared
  with E13's measured Δg: per-net mean in the large and middle ranges, and the correlation across rows.
- **Explains A's reversal** if the predicted mean Δg is > 0 in the large range and < 0 in the middle range in ≥ 5 of 6 A
  nets. SS predicts 0 by construction.
- **Same function in A and B:**
  - F5 is fitted on each group's average g (fit set) and scored on every net's held-out g, within and across groups.
  - "Same" if the across-group R² is ≥ 0.9 x the within-group R², in both directions.
  - Also reported: each group's best named form and the fitted taus.
**E14.3 Causal check (only for a group where a named form fits well).**
- Fresh base episodes: 2,000 new, from W.sample_np(2000, np.random.default_rng(20_261_032)). Identity vectors are drawn as in
  W.tokens_np; then, for each (episode, source) row:
  - two error vectors with the source's actual SS, from the family "m of the 4 errors share a fraction kappa of SS equally,
    the rest share 1 − kappa equally";
  - (m, kappa) drawn independently for each version from m in {1, 2, 3} and kappa in {0.4, 0.6, 0.8, 0.95};
  - random shared signs and positions; everything else identical. Exact Bayes must agree to 1e-8; otherwise stop.
- Before any net sees these pairs, the predicted Δg per row and net from the winning form (fitted in E14.1) is saved and
  its SHA256 logged.
- Measured: Δg on all 12 nets (as above, K = 8). Also reported, with no bar: Δw (effective weight, K = 8).
- **Prediction confirmed** in a group if both hold:
  - in ≥ 5 of 6 nets, the correlation between predicted and measured Δg is ≥ 0.5;
  - the sign agrees on ≥ 70% of the rows whose |predicted Δg| is above its median.
Predictions:
- P97. In group A, the best named form is COUNT_tau, it fits well, and it explains A's reversal.
- P98. The function is not the same in A and B.
- P99. Where E14.3 runs, the advance prediction is confirmed.


### Amendment 21, addendum 1 (2026-10-29 13:27Z): E14.3 predictors, written after seeing E14.1 and E14.2
- Seen:
  - Best named forms: SS in A (held-out R² 0.89 against the learned symmetric function's 0.90) and SA in B (0.52 against
    0.53). Both "fit well" by the registered rule.
  - On the E12 pairs, SS predicts no change by construction; it cannot produce the error-spread effect.
  - COUNT (tau = 0.8 in 11 of 12 nets) and the learned symmetric function reproduce A's sign pattern in 6 of 6 A nets. The
    learned function's predicted Δg correlates 0.71 (A) and 0.34 (B) with the measured Δg; COUNT overshoots in size and
    predicts a reversal in B that B does not show.
- E14.3 runs as registered for both groups with their winning forms:
  - A: SS, which predicts Δg = 0 on every pair. The registered correlation test is undefined for a constant prediction,
    so it is reported as "prediction of no change", checked against the measured |Δg|.
  - B: SA.
- Added (post hoc): frozen predictions from the learned symmetric function (F5) and from COUNT_tau, per net, for both
  groups, judged by the same registered criteria. All predictions are saved and hashed before any net sees the new pairs.


## Amendment 22 (2026-10-29 13:50Z): E15, the error-spread departure in clean transformers; E16 data generation (new agent)

Written by the agent that took over phase B, before any E15 quantity is computed and before any replay is started. Known
beforehand: E12's per-net results on the 12 noisy nets (both ranges, groups); the clean nets' test gaps (tiny-pfn eval.json and
summaries); the binding-plateau account (depth-1 and stalled nets lack functional source binding). Nothing has been measured on
any clean net's pairs, weights or encoding strength.
**Why this episode.** Each of Kelvin's directions (how the nets learn, limits, why not better) depends first on whether the
confirmed departure (E12: a big miss earns more weight than the same SS spread out) belongs to transformers learning this task
or to our noise and sphere architecture. The clean cohort answers that at no training cost: 135 tiny-pfn nets (width 4-32,
1 or 2 heads, depth 1 or 2, attention-only, Gaussian or mixture head, identity dim 4 or 16, regular or long training, 5 seeds
each) plus base_mix4_s5-7. It also contains the 12 noisy nets' own bases, so it tests whether the A/B pattern is inherited.
**Nets and checkpoints** (`e15_data.py` -> `results/e15_nets.json`). Every `tiny-pfn/runs/w*_s[0-9]` run and `runs/base_mix4_s5-7`.
- Best-validation checkpoint (`model.pt`) of every net; also the phase-1 checkpoint (`model_phase1.pt`, best before the
  cooldown) of every acquired net.
- Classes, by the test NLL gap of `model.pt` on the 16,000 test episodes (computed in `e15_run.py`):
  - acquired: depth 2 and gap < 0.10;
  - non-acquired: depth 1, or depth 2 with gap >= 0.10.
- The 12 bases: w32_h2_L2_mix4_s0, s1, s2, s4; w32_h2_L2_mix4_ws_s0-s4; base_mix4_s5-7 (mapped to their noisy nets through
  each noisy net's meta `init_from`).
**Data.** The E12 pairs (`results/e12_pairs.npz`; 2,962 large-range and 3,006 middle-range rows). For ds-4 nets the same
readings and labels with identity vectors redrawn in R^4 (N(0,1)/4, default_rng(20_261_101), one set per base episode, shared by
both versions; `results/e15_pairs_ds4.npz`). Test episodes: the first 4,000 of `testset.npz` / `testset_ds4.npz` (same episodes).
**Measures** (`e15_run.py`; deterministic, no noise draws).
- M1 (primary): effective weights w_s = d m / d o*_s of all three sources (m = predictive mean: the mixture mean, or the
  Gaussian mean for Gaussian heads) in both versions of every pair.
  - Δw = w_src(kappa 0.85) − w_src(kappa 0.25); rom_r = Σ Δw / Σ W over the rows of range r (W = the modified source's Bayes
    weight, `WB0`), as E12.
  - Other sources: rom_other_r = Σ (mean of the two other sources' Δw) / Σ W; specificity spec_r = rom_r − rom_other_r.
- M2: reliability use = OLS slope of w on the Bayes weight W over the first 4,000 test episodes x 3 sources (Bayes = 1).
- M3 (descriptive, no bar): layer-1 encoding strength g = |d h1[QQ_src] / d o*_src| (h1 = residual after block 1, forward
  mode) in both versions; Σ Δg / Σ g(spread) per range.
- Pooled intervals: 1,000 bootstrap resamples of all 3,000 E12 base episodes (empty clusters kept), statistic = mean over the
  nets of each net's mean Δw in the range, as E12. Conditional on the nets.
**Predictions.**
- P100 (the departure exists without noise or sphere normalisation): in the 12 clean bases, large-range rom > 0 in >= 10 of 12,
  and the pooled mean-Δw interval lies above 0.
- P101 (it is general in clean depth-2 nets that bind): large-range rom > 0 in >= 80% of acquired nets.
- P102 (the A/B pattern is inherited from the base): the base's middle-range rom is < 0 for the bases of A nets and > 0 for
  the bases of B nets in >= 10 of 12. Also reported (no bar): the correlation across the 12 pairs of base and noisy rom, each range.
- P103 (it needs binding): |large-range spec| <= 0.05 in >= 80% of non-acquired nets.
- P104 (the cooldown freezes it): over acquired nets, Σ rom_large(final) / Σ rom_large(phase 1) lies in [0.75, 1.33].
- P105 (exploratory, direction only; means over acquired seeds):
  - width: large-range rom is larger at width 16 than at 32 in >= 2 of 3 matched comparisons (w16_h1_L2 vs w32_h1_L2;
    w16_h2_L2 vs w32_h2_L2; w16_h2_L2_ds4 vs w32_h2_L2_ds4);
  - training length: smaller in the long arm in >= 2 of 3 (w16_h1_L2_long vs w16_h1_L2; w32_h2_L2_long vs w32_h2_L2;
    w16_h2_L2_ds4_long vs w16_h2_L2_ds4).
**What the outcomes would mean.**
- P100 and P101 hold: the departure is a property of how these transformers learn the task, not of the noise or the sphere;
  the dynamics episode (E16) can then study it in the clean base training.
- P100 fails: it is created by the noisy fine-tune (architecture change, noise, or the extra training); E16 then centres on
  the fine-tune stage.
- P102 holds: the A/B split is set before the noise fine-tune; fails: the fine-tune creates or reshuffles it.
**E16 data generation (no analysis registered here).**
- Replays of base_mix4_s5 (whose fine-tune is the B net L2nb5) and base_mix4_s6 (A net L2nb6) through `train_ckpt.py base
  --seed S --out runs/dyn_base_sS`. The recipe's code is unchanged; a subclass of AdamW saves the state dict at step 0, every
  100 steps to 2,000, every 250 to 5,000, every 500 to 10,000 and every 1,000 after that.
- Determinism check: the replay's validation curve must equal the original `meta.json` curve (to 1e-9) and its `model.pt`
  must equal the original bitwise. If not, the checkpoints are reported as a sibling trajectory.
- No checkpoint is analysed before a later amendment registers the E16 measures and predictions. The training log
  (validation NLL every 1,000 steps) is seen while it runs.


## Amendment 23 (2026-10-29 14:11Z): E16, training dynamics of the clean bases (one A and one B lineage)

Written before any checkpoint of the replays is measured. Known beforehand:
- E15 in full, including the endpoints of these lineages (best checkpoints): base_mix4_s5 (B lineage) large-range rom +0.158,
  middle +0.064; base_mix4_s6 (A lineage) +0.078 and −0.037.
- E15 post hoc: among two-head nets, A-type (negative middle range) nets are closer to Bayes, and the learning-rate decay moved
  3 nets from B-type to A-type, none back.
- The replays' validation curves (they match the originals at every logged step so far).
**Why this episode.** Kelvin asked how the nets learn, including what happens over training. E15 places the departure and the
group signature in the clean base training, so the base trajectories are where they form. The A-type-is-later hypothesis
(post hoc in E15) makes a sharp prediction about the A lineage's path.
**Data.** Every saved checkpoint of `runs/dyn_base_s5` and `runs/dyn_base_s6` (step 0; every 100 to 2,000; 250 to 5,000; 500
to 10,000; 1,000 to 90,000). The determinism check of amendment 22 is reported; if it fails, the checkpoints are a sibling
trajectory and the endpoints are re-measured from the replay itself.
**Measures** (`e16_run.py`, deterministic; E15's definitions):
- gap (16,000 test episodes); slope of the effective weights on the Bayes weights (first 4,000 test episodes);
- on the E12 pairs: rom and spec in both ranges; total-weight change tot_r = Σ ws1 / Σ ws0 − 1 over the three sources;
- the normalised encoding surrogate Δg_sph/g by range (E15 post hoc definition: uncentred sphere of h1);
- reliability code: held-out R² of g_sph (all sources pooled) from a piecewise-linear function of log SS (knots at the fit
  deciles; fit on test episodes 0-1,999, scored on 2,000-3,999);
- deviation pattern: predictive mean minus the Bayes mean on the first 4,000 test episodes.
**Events** (`e16_summ.py`): t_acq = first checkpoint from which the gap is < 0.10 at three consecutive checkpoints; t_dep = first
checkpoint from which spec_large > 0.05 at three consecutive checkpoints; plateau band = checkpoints before t_acq with gap in
[0.14, 0.20]; "final" = the original best step (s5: 89,000; s6: 90,000).
**Predictions.**
- P106 (the source-specific departure is born with acquisition): |t_dep − t_acq| ≤ 500 steps in both lineages.
- P107 (a non-specific precursor before acquisition): with ≥ 3 checkpoints in the plateau band, the band's median tot_large is
  > +0.02 and its median spec_large is < 0, in both lineages (as in E15's non-acquired nets).
- P108 (larger earlier): in both lineages the maximum large-range rom after t_acq exceeds 1.33 x the final value, and the
  Spearman correlation of large-range rom with the step over checkpoints at or after t_acq + 1,000 is < −0.5.
- P109 (A is a later stage): the A lineage's middle-range rom is > 0 at the first three checkpoints at or after t_acq + 1,000
  and < 0 at the final checkpoint; the B lineage's is > 0 at >= 90% of the checkpoints at or after t_acq + 1,000.
- P110 (the encoding surrogate tracks the departure): Spearman(Δg_sph/g large, rom large) > 0.5 over checkpoints at or after
  t_acq, in both lineages.
- P111 (shared early, more seed-specific late): the Pearson correlation of the two lineages' deviation patterns at matched
  steps, averaged over steps in [T + 1,000, T + 5,000] with T = the later t_acq, exceeds its value at step 90,000 by >= 0.1.
**Also reported (no bars):** the curves; the reliability-code R² over training; when each lineage's deviation pattern first
correlates > 0.8 with its final one; the large-range rom against the gap along each trajectory after acquisition (to compare
with E15's cross-sectional relation); whether either lineage meets the binding paper's plateau definition.
**What the outcomes would mean.**
- P106 holds: the departure is part of the reliability computation from the moment binding forms, not a later add-on.
- P107 holds: the error-spread sensitivity predates binding (as E15's non-acquired nets suggest), and binding routes it.
- P108 holds: training slowly corrects the departure, without removing it; fails: it is set at acquisition and frozen.
- P109 holds: the A-type signature is a later refinement of a B-type solution; fails on the first part: A is A from the start.

### Amendment 23, addendum 1 (written after the coordinator set target M2; no change to E16's plan, measures or bars)
- Link to M2 (a general account of how transformers approximate Bayes in context and why they deviate): E16 asks when the
  departure forms relative to source binding and whether training removes it. If it is born with the reliability computation
  and only slowly corrected, the inductive bias acts at acquisition; that constrains the "why" (E19's minimal model) and tells
  the generality tests (E17, E18, E20) to measure the departure at the end of training as a stable property.
- At this point about 180 of the roughly 240 checkpoints had been measured by `e16_run.py`, and nothing from them had been
  read (only the run logs, which print each checkpoint's gap).


## Amendment 24 (2026-10-29 14:48Z): E17, does depth remove the departure where extra attention capacity does not? (sizes, existing nets)

Written before any E17 quantity is computed. Known beforehand: E15 and E16 in full; the gaps of these nets (meta/eval files). None
of the nets below has been measured on the pairs or the test weights.
**Link to M2.** M2 asks for an account that holds across sizes and predicts the departure from the architecture. This episode
tests a prediction derived from the architecture, not fitted, on larger nets that already exist.
**The hypothesis being tested (pooling before nonlinearity).**
- In a 2-layer net, a source's reliability must be gathered by its query-question token in layer 1. The only way the final
  read-out (layer 2, query token) can weight each source's reading by that source's own errors is if the source's query token
  has already pooled them. That layer-1 pooling acts on the labelled tokens' embeddings, before any per-token MLP. The only
  nonlinearity there is LayerNorm, which makes every per-token error feature bounded: roughly quadratic for small errors and
  saturating for large ones. A bounded feature undercounts one big miss relative to its square, which is the observed sign at
  large errors.
- With three layers, a labelled token can square its own error in layer 1's MLP, the query-question token can pool those
  squares in layer 2, and the read-out can happen in layer 3. Depth removes the constraint; extra layer-2 heads or wider layer-2
  heads do not.
- E16 shows that the departure is born with binding, so a 3-layer net may still first find the layer-1 solution. The prediction
  is about the end of training.
**Nets** (`results/e17_nets.json`, written before this amendment; regenerated to this list).
- The clean heads cohort (`tiny-pfn-interp/heads`; pfn2.PFN), 5 seeds each, Gaussian (g) and warm-started mixture (ws) heads:
  - base: 2 layers, 2 heads;
  - h3, h4: 3 or 4 layer-2 heads of dimension 16;
  - w64: 2 layer-2 heads of dimension 32;
  - L3: 3 layers.
- The nets closest to Bayes: L3_s0.0 (3 layers, sphere architecture, no noise; validation gap 0.0003), L3_s0.05 and L2_s0.0
  (validation gaps 0.008 and 0.003). Their clean bases are L3_ws_s13 (in the cohort) and w32_h2_L2_mix4_ws_s4 (E15).
**Measures** (`e17_run.py`): as E15 (rom, spec and total-weight change in both ranges on the E12 pairs; slope on the Bayes
weights; test gap). The noisy net at sigma 0.05 uses noise-averaged effective weights (K = 8 draws, E12's draw order), the gap from
the K = 4 noise-marginalised mixture; sigma 0 nets use one pass. Acquired = gap < 0.10.
**Predictions.**
- P112 (depth shrinks it strongly): the mean large-range rom of the acquired L3 nets is < 0.5 x that of the acquired base nets
  of the same cohort.
- P113 (more layer-2 attention does not): the mean large-range rom of the acquired h3, h4 and w64 nets is >= 0.75 x the base
  nets' mean, and rom > 0 in >= 80% of them.
- P114 (the nearest-to-Bayes 3-layer net has almost none): L3_s0.0's large-range rom is < 0.03.
- Reported without a bar: L2_s0.0 and L3_s0.05 (whether a 2-layer net close to Bayes keeps it; the 3-layer net with noise).
**What the outcomes would mean.**
- P112 and P113 hold: the departure is a depth-limited consequence of pooling before any per-token nonlinearity; the
  architecture predicts which changes remove it.
- P112 fails: depth does not help, so the bias is not the layer-1 pooling constraint, or gradient descent keeps the first
  (layer-1) solution even when a better path exists.
- P113 fails: layer-2 attention capacity matters, against the hypothesis.


## Amendment 25 (2026-10-29 15:03Z): E18, do pretrained LLMs show the big-miss departure in context?

Written before the items are generated and before any experimental call. Known beforehand: E12-E17; a plumbing check with the
prompt "Reply with the number 1.23" (direct Messages API calls through the proxy were refused upstream with 429 for all three Claude
models; the Claude Code CLI in print mode and `codex-any exec` both worked).
**Link to M2.** M2 asks whether the account holds across pretrained LLMs. If LLMs that read a source's past errors in context also
give a source with one big miss more influence than one with the same squared total spread out (where the stated model makes the
two equivalent), the departure is a property of transformer in-context aggregation beyond our small nets; if not, it is specific to
small nets trained on this prior.
**Access** (`e18_llm.py`): Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 through `claude -p` (bare, no tools, no session persistence,
no setting sources, JSON output); GPT-6 (gpt-6-astra) through `codex-any exec`. No sampling parameters (not accepted); effort is
the only setting. A GPT call that used a tool, or any call without a parsable number, counts as failed.
**Items** (`e18_data.py` -> `results/e18_items.json`, generated after this hash and then hashed):
- 400 fresh episodes from W.sample_np(400, default_rng(20_261_105)); per episode, the first source with SS >= 3.182394 (E12's large
  range) and |reading - Bayes mean| >= 0.75; the first 60 such episodes.
- Matched versions of that source's four errors: big miss (kappa 0.85) and spread (kappa 0.25), same SS, signs and position (E12's
  construction, random signs and position from the same generator); the other sources keep their errors.
- All displayed numbers rounded to two decimals; Bayes mean and weights computed from the displayed numbers (so the two versions'
  Bayes means differ only by rounding, reported as "Bayes shift").
- Source labels A, B, C in a random order per episode (same in both versions).
- Two framings, verbatim in `e18_data.py`: "stated" (the generating model is described: Y ~ N(0, 1), Gaussian errors with a
  per-source variance from {0.1, 0.5, 2.0}) and "unstated" (forecasters, no model).
**Arms** (`e18_call.py`): stated framing at low effort for Haiku, Sonnet and Opus (60 pairs, 2 samples per prompt) and GPT-6 (60
pairs, 1 sample); Opus at high effort (stated, first 30 pairs, 1 sample); Sonnet at low effort in the unstated framing (60 pairs, 1
sample). About 1,020 calls. Calls are sequential per harness, at most two harnesses at once, 1 s apart; an arm stops after 10
consecutive failures. Spend cap: stop if the Claude list-price total passes $25 or the calls pass 1,200.
**Measures** (`e18_summ.py`):
- Per pair: the mean answer over samples in each version, m_big and m_spread; lever L = reading of the modified source minus the
  Bayes mean (spread version); W = its Bayes weight (spread version).
- shift = Σ (m_big − m_spread) · sign(L) / Σ W · |L|, E12's prediction-shift statistic: the extra pull toward the modified source's
  reading in the big-miss version, in units of Bayes's own pull. Bayes: 0 up to rounding. 95% intervals from 1,000 bootstrap
  resamples of pairs.
- Reliability use: OLS slope of (m − mean reading) on (Bayes mean − mean reading) over all answered prompts (Bayes = 1; equal weights
  = 0). A model is "evaluable" for the departure if this slope is >= 0.3 (otherwise it is not using the error histories enough for
  the pair contrast to mean much).
- Also reported: pairs with positive and negative contrast, mean absolute deviation from Bayes, within-prompt sd (2-sample arms),
  calls, failures, cost and tokens.
**Predictions.**
- P115 (the departure appears in pretrained LLMs): among the evaluable stated-framing low-effort models, the shift is > 0 in all but
  at most one, and the pooled shift (mean over evaluable models, bootstrap over pairs) has an interval above 0. Untestable if fewer
  than two models are evaluable.
- P116 (reasoning reduces it): Opus at high effort has |shift| < 0.5 x Opus at low effort on the same 30 pairs.
- P117 (an unstated model enlarges it): Sonnet's shift is larger in the unstated framing than in the stated one.
**What the outcomes would mean.**
- P115 holds: LLMs, given a model under which both histories are equivalent, still let a source with one big miss move their
  estimate more; the departure is general across transformer aggregators in context (its cause in LLMs may be their text prior,
  which this episode does not separate).
- P115 fails with evaluable models: the departure is specific to small nets trained on this prior.
- P116 holds: explicit reasoning moves toward the sum-of-squares statistic; the departure belongs to fast in-context aggregation.

### Amendment 25, addendum 1 (written after the first 2 GPT calls and 7 Haiku calls; before any analysis)
- Seen: GPT-6's first two calls (pair 29, spread version) used a tool, presumably code to compute the answer. Both were excluded
  as failures, as registered, and the GPT stream was stopped. Claude arms cannot use tools (`--tools ""`).
- Change (equalises the conditions): GPT calls now run with `--disable shell_tool --disable code_mode_host --disable browser_use
  --disable computer_use`. Checked with a non-experimental prompt asking it to run `echo`: it replied that it cannot run commands.
  The GPT arm resumes with this setting; the two excluded calls stay in the log.
- Also seen, no change: Claude Haiku 5.5 at low effort uses about 6,300 thinking tokens per call (about 28 s). "Low effort" is the
  registered setting, not "no reasoning"; the arms are read as such.


## Amendment 26 (2026-10-29 15:09Z): E19, the isolating intervention: a per-token MLP before the first attention

Written before any E19 net is trained. Known beforehand: E15-E17; E18 is collecting LLM answers in parallel (no E18 result has been
looked at). Plumbing check: with seed 5 the variants' shared parameters equal the baseline replay's step-0 weights exactly.
**Link to M2.** M2 asks for an account that predicts the departure from the architecture. E17 found depth reduces the departure
where extra last-layer attention does not, consistent with the account that a 2-layer net must pool each source's errors before any
per-token nonlinearity other than LayerNorm (which bounds each error's contribution, so a big miss is undercounted). E17 could not
separate this from generic depth effects. This episode changes only the pooling order.
**Design** (`pfn_var.py`, `train_var.py`): the tiny-pfn recipe (pfn.train: width 32, 2 heads, depth 2, mixture head K = 4,
AdamW 1e-3, batch 256, 40k/80k constant phase by the convergence rule, 10k cooldown), unchanged, on two variants with one extra
token-wise MLP block (pre-LN, residual, hidden 64; 4,256 parameters):
- "pre": the block runs on every token before the first attention, so each labelled token can transform its own reading and label
  (for example square its error) before the source's tokens are pooled;
- "post": the same block after the last block, where it cannot change what is pooled.
Seeds 5, 6 and 7, the seeds of base_mix4_s5-7 (the baselines; E15: large-range rom +0.158, +0.078, +0.157). With the same seed, the
shared parameters start identical to the baseline's and the training data stream is the same; only the extra block differs.
Checkpoints as train_ckpt.py.
**Measures** (`e19_run.py`, `e19_summ.py`): E15's on the best checkpoint (rom, spec, gap, slope on the Bayes weights, pooled
mean-Δw interval over base-episode bootstraps). Acquired = test gap < 0.10; only acquired nets and seed pairs whose baseline and
variant are both acquired enter the tests (at least two pairs needed).
**Predictions.**
- P118 (pre-MLP removes most of it): the pre nets' mean large-range rom is < 0.5 x the mean of their same-seed baselines, and it is
  lower than the baseline in every seed pair.
- P119 (post-MLP does not): the post nets' mean large-range rom is >= 0.75 x the baselines' mean.
**Also reported (no bars):** gaps (an extra block may also lower the gap); the middle range; the large-range rom over the
checkpoints of the pre nets at seeds 5 and 6 against E16's baseline trajectories of the same seeds.
**What the outcomes would mean.**
- Both hold: the departure is caused by pooling each source's errors before any learned per-token nonlinearity; changing only the
  pooling order removes most of it, so the architecture predicts it.
- P118 fails: the pooling order is not the main cause, or gradient descent does not use the pre-pooling block to transform the
  errors; E17's depth effect then needs another explanation.
- P119 fails: extra capacity anywhere reduces it, and the pooling-order account loses its specificity.

### Amendment 25, addendum 2 (pacing only; written before any analysis)
- Seen: Haiku calls take about 23 s and GPT-6 calls about 100 s (no tools). One sequential Claude chain would take most of a day.
- Change: the Claude arms run as two chains at once (Haiku -> Sonnet -> unstated Sonnet; Opus low -> Opus high), so at most two
  concurrent Claude calls, plus the GPT stream. Items, prompts, settings, measures and bars are unchanged.


## Amendment 27 (2026-10-29 15:14Z): E20, a minimal model of the reliability computation, with signs predicted from LayerNorm

Written before any E20 model is trained (a 200-step smoke test wrote only to /tmp). Known beforehand: E15-E17; E18 and E19 are
running and none of their results has been looked at.
**Link to M2.** M2 asks for an account that explains the inductive bias and predicts the departure from the architecture. This
isolates the transformer's layer-1 constraint in a model small enough to analyse, and derives the sign before training.
**The argument (before any data).** In a 2-layer transformer, a source's labelled tokens are pooled before any per-token MLP, so
each token's error reaches the pool only through LayerNorm of its linear embedding. If the embedding is c0 + a r u + n (constant,
error and nuisance parts), the normalised token's component along c0 is C / sqrt(C^2 + a^2 r^2 + |n|^2): bounded, decreasing in
r^2 and convex in r^2. A sum of a convex function of the squared errors is smallest when a fixed sum of squares is spread evenly.
So a pooled LayerNorm feature reads spread errors as the least reliable and a single big miss as more reliable, at every error
level: a positive departure (group B's sign) in both ranges. A per-token MLP before pooling can compute r^2 itself, which removes
the constraint. Noise added after LayerNorm should push the learned embedding toward saturation (a larger error component is needed
to beat the noise), which enlarges the departure.
**Design** (`minimal.py`, `e20_summ.py`): one source's v from its four labelled errors (3-way cross-entropy; Bayes depends on SS
only), tokens carrying identity vectors and labels as in the PFN. Models: ln (linear embedding -> LayerNorm -> attention pool ->
readout MLP), pre (linear -> per-token MLP -> LayerNorm -> attention pool -> readout), ss (oracle log SS -> readout). ln and pre at
sigma 0 and 0.1 (noise after LayerNorm), ss at sigma 0; seeds 0, 1, 2; 30,000 steps of AdamW (1e-3, weight decay 0.01), batch 512.
**Measure.** On 4,000 fresh single-source episodes (default_rng(20_261_107)) in E12's large and middle ranges, matched big-miss
(kappa 0.85) and spread (kappa 0.25) error vectors with shared labels and identity vectors; predicted posterior precision
P = sum_k p_k / v_k; departure = sum (P_big - P_spread) / sum P_Bayes per range (noise-averaged over 16 draws when sigma > 0).
**Predictions.**
- P120 (the sign follows from LayerNorm-then-pool): the ln model at sigma 0 has a positive departure in both ranges in all three
  seeds.
- P121 (a per-token MLP before pooling removes most of it): the pre model's mean large-range departure has absolute value < 0.5 x
  the ln model's (sigma 0).
- P122 (noise enlarges it): the ln model's mean large-range departure is larger at sigma 0.1 than at sigma 0.
- Check (no bar): the ss oracle is near 0.
**What the outcomes would mean.** P120-P122 hold: the departure's sign and its dependence on pooling order and noise follow from
LayerNorm geometry alone, in a model with nothing else. P120 fails in the middle range only: error-dependent attention weights can
reverse the middle-range sign (a candidate for group A). P121 fails: the bias is not the pooling order.

### Amendment 25, addendum 3 (answer parsing; written after a format check, before any outcome analysis)
- Seen (`e18_format.py`): every Haiku and GPT answer so far is a bare number; 85 of 96 Opus answers are; the other Opus answers
  mostly begin with the bolded number and then explain, but one begins with its working ("...the sum of squares is 0.0361 + ..."),
  where the registered rule (the first number in the text) would take a wrong number.
- Change: the answer is the number the response begins with (after optional bold markers and spaces); a response that does not
  begin with a number counts as non-compliant (excluded), and its item is asked again when the arm is resumed. Stored answers are
  re-parsed from their saved text under this rule (`e18_llm.parse`, used by `e18_call.py` and `e18_summ.py`).
- The streams already running keep the old in-memory parser for their own retry decisions; their records are re-parsed in the
  analysis, and every arm is resumed once after its stream ends to re-ask any non-compliant item.

### Amendment 27, addendum 1 (E20.2; written after seeing E20's registered results, before running anything new)
- Seen: the ln model (attention pooling) has a positive large-range departure in all seeds (+0.039) but a negative middle-range
  departure (−0.062), so P120 fails in the middle range only, the outcome the amendment said would point to error-dependent
  attention weights. pre removes most of it (P121 holds); noise enlarges it (P122 holds).
- E20.2 (registered now): the same ln model with the attention replaced by a plain mean over the four tokens ("lnmean"), sigma 0
  and 0.1, seeds 0-2, otherwise identical. The pure LayerNorm argument predicts a positive departure in both ranges.
- P123: lnmean at sigma 0 has a positive departure in both ranges in all three seeds, and its mean middle-range departure is above
  the attention model's (−0.062).

### Amendment 27, addendum 2 (E20.3; written after E20/E20.2 results and the review, before running anything new)
- The review noted that the attention explanation of group A's signature rests on separately trained models and a probe that
  changes SS. E20.3 intervenes on attention in the trained ln models themselves (`e20_freeze.py`): evaluate the same matched pairs
  with the attention weights replaced by uniform weights (1/4), keeping embedding, LayerNorm, noise and readout fixed.
- P124: for the three ln models at sigma 0, uniform weights make the middle-range departure positive and the large-range departure
  larger than with the learned attention, in all three. (Caveat registered now: uniform pooling feeds the readout vectors it was
  not trained on, so a failure could also mean the readout breaks off-distribution.)

### Amendment 25, addendum 4 (answer parsing for responses that end with the answer; before any outcome analysis)
- Seen: Sonnet 5.5 (stated and unstated framings, low effort, no thinking tokens) writes its working first and ends with the answer
  alone in bold on the last line. Under addendum 3 all its responses were non-compliant, so S_sonnet stopped after 10 failures
  and U_sonnet after 10. Stored texts were truncated to 300 characters. A plumbing check on a dummy prompt in the same format
  (numbers from default_rng(999), not a registered item) at low and medium effort shows the same format.
- Change: the answer is the number the response begins with (addendum 3); otherwise the number that alone makes up its last
  non-empty line (optionally bold); otherwise non-compliant. New calls store the full text. Earlier records are re-parsed; a
  truncated non-compliant record stays non-compliant and its item is asked again. S_sonnet and U_sonnet are resumed.
- Reading: in every arm the Sonnet answers come after explicit visible working, so the Sonnet arms are read as low-effort
  answers with visible reasoning, not as fast intuition.


## Amendment 28 (2026-10-29 16:59Z): E21, does the departure persist, shrink or flip with width? (GPU, approved by the coordinator)

Written before any E21 net is trained. Known beforehand: E15-E17, E20 in full; E18 and E19 are still running (none of their
outcomes looked at). GPU approved by the coordinator (on-demand L4, cap $10, every VM in `GPU-LOG.md`, deleted at the end).
**Link to M2.** M2 asks for an account that holds across sizes and predicts the departure from the architecture. The account from
E17/E20: in 2 layers each source's errors are pooled after LayerNorm and before any per-token MLP, which caps each error's
evidence; error-dependent attention partly compensates (group A's pattern is an imperfect correction). Width does not remove the
pooling order, so the account predicts the departure persists at any width; more capacity may compensate better, so it may shrink.
**Design** (`gpu/gpu_e21.sh`, `gpu/launch_e21.sh`, `e21_summ.py`): the tiny-pfn recipe through `pfn2.py` (identical to pfn.py's
training loop apart from the device): 2 layers, 2 heads, mixture head K = 4, AdamW 1e-3, batch 256, the 40k/80k convergence rule
and 10k cooldown; widths 64, 128 and 256 (head dimension width/2), seeds 0-4 (15 nets), one L4 VM, up to 6 trainings at once,
timeout 3 h. Width 32 is E15's w32_h2_L2_mix4 nets (mix4 s0-s4 and base_mix4_s5-7). Measures as E15 (rom, spec, test gap,
pooled mean-Δw interval); acquired = test gap < 0.10; the A/B signature = the sign of the middle-range rom.
**Predictions.**
- P125 (persists with width): at widths 128 and 256, large-range rom > 0 in >= 80% of the acquired nets, and their mean is
  >= 0.05 at each width.
- P126 (shrinks with width): over the acquired nets of widths 32, 64, 128 and 256, Spearman(width, large-range rom) < 0, and the
  width-256 mean is below the width-32 mean.
- P127 (both group signatures still form): among the acquired nets at widths 128 and 256, at least one has a negative and at least
  one a positive middle-range rom.
**What the outcomes would mean.** P125 holds: width does not remove the departure, as the pooling-order account says (unlike depth,
E17). P125 fails: width alone removes most of it, so the cap is a capacity limit rather than a structural one. P127 fails (one
sign only): at scale the nets settle on a single solution type.


## Amendment 29 (2026-10-29 17:48Z): E22, the toy's attention intervention inside the real nets

Written before any E22 quantity is computed. Known beforehand: E15-E17, E19-E20 in full (E20.3: in the toy, making the attention
uniform restores a large positive departure in both ranges); a plumbing check on w32_h2_L2_mix4_s0 (200 pair episodes): the
re-implemented block 1 matches the model exactly; the patch changes predictions; within-source attention entropy 1.02 against
1.39 for uniform, and 98.6% of the query-question token's layer-1 attention falls on its own source's labelled tokens.
**Link to M2.** M2 asks for an account that explains the inductive bias in the real transformers. E20 proposed: pooled LayerNorm
features cap each error's evidence, and error-weighted attention is an imperfect correction (group A's pattern). This tests whether
the real nets use that correction.
**Design** (`e22_run.py`, `e22_summ.py`): every acquired depth-2, ds-16 clean net of E15 (`results/e22_nets.json`, 48 nets:
1- and 2-head, widths 16 and 32, with and without MLP, both head types). In block 1, for each source's query-question token and
every head, the attention weights on that source's four labelled tokens are replaced by their mean (uniform within the group, same
total mass); everything else unchanged. Effective weights on the E12 pairs, unpatched and patched (autograd, deterministic); rom in
both ranges as E12.
**Predictions** (from the toy).
- P128: the patch raises the large-range rom in >= 80% of the nets.
- P129: the patch raises the middle-range rom in >= 80% of the nets, and among the nets whose unpatched middle-range rom is
  negative (A-type), it turns positive in >= 2/3.
**Caveat registered now.** The patched attention is off-distribution for everything downstream, so changes could reflect general
disruption rather than the removal of a specific correction; a failure means the real nets do not rely on within-source weighting
the way the toy does.


## Amendment 30 (2026-10-29 17:51Z): E23, a layer-1 attention signature of the A/B groups, tested on E21's fresh GPU nets

Written after E22's registered results and its post-hoc attention analysis (`e22_post.py`), before any E21 net has been measured
(the E21 job is still running on the GPU; nothing from it has been read except training-log lines).
**Seen post hoc (E22).** In layer 1, the query-question token's attention share over its own source's four labelled tokens, as a
function of each token's |error| (Spearman rho over 4,000 test episodes): all 6 group-A bases have rho -0.08 to -0.09; 5 of 6 group-B
bases have rho +0.06 to +0.10. Over 30 two-head clean nets: rho > 0 goes with a positive middle-range rom (B) in 9 of 9; overall sign
agreement 24 of 30. All 18 one-head nets have rho < 0 and are B-type.
**Link to M2.** The A/B split is part of the account (E15: it is set in the clean base; E16: A comes later; E20: in the toy, A is an
attention-based correction). A layer-1 attention signature that predicts the group on fresh nets would locate the split
mechanistically.
**Design** (`e23_run.py`): every acquired E21 net (2 heads; widths 64, 128, 256): rho as in E22's post-hoc analysis; compared with
the net's middle-range rom from E21's measures.
**Prediction.**
- P130: every acquired E21 net with rho > 0 has a positive middle-range rom, and the signs of rho and of the middle-range rom agree
  in >= 80% of the acquired E21 nets.


## Amendment 32 (2026-10-29 18:29Z): E18b, single-pass LLM answers (no reasoning) on zero-mean matched pairs (coordinator's request)

Written before the E18b items are generated and before any experimental call. Known beforehand: E18 in full (reasoning LLMs
given the model track Bayes; no departure detected; Sonnet's unstated-framing shift confounded with bias handling). Plumbing checks
on a dummy prompt (default_rng(999), not an item): with thinking off (MAX_THINKING_TOKENS=0, alwaysThinkingEnabled false), a 16-token
output cap and the system instruction SYS, Claude Haiku 4.5, Sonnet 4.6 and Opus 4.8 answer with a bare number and 0 thinking
tokens; Sonnet 5.5 refuses (safeguard) and Haiku 5.5 cannot switch thinking off; GPT-6 has no effort below "low", gpt-5.6-sol accepts
"none"; Qwen3-1.7B runs locally (non-thinking template, greedy) at about 13 s per prompt. (Amendment 31's number is reserved for E24.)
**Link to M2.** Does the small nets' pooling bias reappear in a single forward pass of large pretrained models, so that reasoning
is what fixes it in E18?
**Items** (`e18b_data.py` -> `results/e18b_items.json`, generated after this hash and then hashed): E18's 60 base episodes; the
modified source's errors rebuilt so that both versions keep its SS and have zero mean error: big miss (kappa 0.75, one error with a
random sign, three at a third of its size with the opposite sign) and spread (two positive and two negative errors of equal size).
Bayes from the displayed (rounded) numbers. Both E18 framings.
**Arms** (`e18b_call.py`): S_ and U_ (stated, unstated) x haiku45, sonnet46, opus48 (claude -p, thinking off, 16-token cap, SYS
appended), gpt56 (codex-any, gpt-5.6-sol, reasoning none, tools disabled, SYS prepended), qwen17 (local Qwen3-1.7B, greedy, SYS as
system message). Two samples per prompt for the stated API arms, one otherwise. Answer = the leading number; non-compliant
responses are retried up to 3 times. API spend cap $10 at list price (stop the arms if passed).
**Measures** (`e18b_summ.py`): E18's shift, reliability slope (evaluable if >= 0.3), mean absolute deviation from Bayes;
bootstrap over pairs; pooled shift = mean over evaluable models of a framing.
**Predictions.**
- P131 (the nets' direction reappears without reasoning; the coordinator's hypothesis): stated framing, the pooled shift over the
  evaluable single-pass models has an interval above 0, and the shift is > 0 in all but at most one of them.
- P132 (with mean error held fixed, single-pass models distrust a big-miss forecaster when no model is given): unstated framing,
  the pooled shift over evaluable models has an interval below 0.
**What the outcomes would mean.** P131 holds: without reasoning, large pretrained models show the small nets' big-miss tolerance,
and E18's agreement with Bayes came from reasoning. P131 fails (null or negative): the small nets' bias does not appear in single
passes of these models either. P132 decides whether E18's unstated-framing shift survives once bias correction is ruled out.


## Amendment 31 (2026-10-29 18:37Z): E24, on-distribution transplants that split the big-miss weight difference by route

Written before E24 is run on the registered nets. Known beforehand: E15-E23 (E22: making block 1's within-source attention uniform
disrupts the nets and reverses the departure; E23 pending at registration time of this text was then confirmed). Plumbing check
(disclosed): on w32_h2_L2_mix4_s0, the first 50 pair rows, the self-transplant (spread into spread) is an exact identity, and the
printed mean weights were ref 0.194, att 0.207, val 0.212, both 0.187, spread 0.187.
**Link to M2.** M2 asks for the inductive bias in the real transformers. E22's intervention was off-distribution. Here every
replaced quantity comes from the matched spread version of the same episode, so it is a natural value: this splits the big-miss
weight difference between how block 1 weights the source's labelled tokens (attention row of its query-question token) and what
those tokens carry (their block-1 values), and bounds the rest (other routes, e.g. layer 2 reading the labelled tokens directly).
**Design** (`e24_run.py`, `e24_summ.py`): E22's 48 acquired clean nets. For every E12 pair, the big-miss version is run with
(att) the attention row of the modified source's query-question token, all heads, from the spread version; (val) the block-1
values of that source's four labelled tokens from the spread version; (both). The attention row of that token is computed without
gradient in every variant including the reference, so the weights compare like with like. removed(X) = Σ (w_ref − w_X) ÷
Σ (w_ref − w_spread), large and middle ranges, pooled over nets and per net; also for the 6 A and 6 B bases.
**Predictions.**
- P133 (block 1 carries it): pooled over the 48 nets, transplanting both removes >= 70% of the large-range weight difference.
- P134 (the values carry more than the weighting, as the capped-evidence account expects): pooled, the value transplant removes
  more of the large-range difference than the attention-row transplant.


## Amendment 33 (2026-10-29 18:54Z): E25, is the contrast made at the source's own pooling, and by which terms?

Written before E25 is run. Known beforehand: E24 in full and its review (the value transplant there changed every reader of the
source's labelled tokens, so it did not isolate the source's own pooling). Plumbing on w32_h2_L2_mix4_s0, 20 rows: the re-implemented
forward equals the model, and a self-transplant is an exact identity (no weights computed).
**Link to M2.** Locating the departure to one computation (the source's query-question token pooling its labelled tokens) is what an
architecture-predictive account needs; E24 could not separate that from other readers.
**Design** (`e25_run.py`, `e25_summ.py`): E22's 48 nets, E12 pairs. In block 1, at the modified source's query-question token only,
the terms a_t v_t of its four labelled tokens are replaced by the spread version's terms (its attention weights times its values);
all other tokens' attention outputs and all other terms unchanged. Variants: qq_all (four terms), qq_big (the term at the big-error
position, i.e. the largest |error| in the big-miss version), qq_small (the other three). The attention row of that token is computed
without gradient in every variant and in the reference (as E24). removed(X) as E24, large range, pooled over nets; A and B bases.
**Predictions.**
- P135 (the contrast is made at the source's own pooling): qq_all removes >= 70% of the pooled large-range contrast.
- P136 (the big-error token's term carries more of it than the three small ones): qq_big removes more than qq_small, pooled.

### Amendment 32, addendum 1 (pacing only; before any outcome analysis)
- Sonnet 4.6 single-pass calls are slow (about 18% exceed the 16-token cap and are retried). The Opus 4.8 arms now run as a second
  Claude chain alongside the Sonnet arms (at most two concurrent Claude calls). Items, prompts, settings, measures and bars unchanged.


## Amendment 34 (2026-10-29 20:35Z): E26, the error budget by scale: is the big-miss departure part of the cross-source tangledness?

Written before any E26 quantity of a registered net is computed and before the registered data file exists. Known beforehand:
E1-E25 and E18b in full; the theory lane's note `outbox/scale-free-theory-2026-10-29.md` and its files, including N5 (40 clean
tiny-PFN nets: the cross-source interaction share of the net's log-odds correlates 0.96 with KL; I read its stored deviation shares
for a few nets, e.g. w32_h2_L2_mix4_s0's deviation has a source-vs-rest interaction share of 0.45 in N5's units, which suggested
that about 40% of that net's within-context deviation variance is cross-source) and N6 (12 noisy nets: cross-source share lower in
group A, 6 of 6 below B's median; correlation 0.945 with KL).
**Plumbing checks (disclosed; no registered net evaluated).**
- Estimators on planted synthetic functions (seeds 777-779, 6,000 bases, `e26_check.py`): context, own-sufficient, own-pattern,
  tangled-sufficient and tangled-pattern variances recovered within ±16% (no consistent bias over the three seeds); the 2x2
  tangledness estimate is exactly 0 for an additive function with large main effects (the pick-freeze estimate there is off by
  about 1% of the total variance); the KL-exact pattern cost of planted Gaussian shifts recovered (0.01995 against 0.02000).
- `e26_run.py` and `e26_summ.py` run end to end on a random-init net with check data (seed 777, 500 bases); the summary's code
  paths for M, L1 and DY were exercised with copies of that output.
- Reference floors on the check seed (properties of Bayes, no net): Bayes's moment-matched Gaussian has KL 0.0099 and a log-odds
  deviation variance of 0.98 (six deltas), about half of it cross-source; Bayes reduced to 4 components (Runnalls merging) has KL
  0.0001 and deviation variance 0.025, mostly cross-source. So a 4-component head is not forced into a large KL, but the log-odds
  metric is far more sensitive to the posterior's shape than KL is.
**Link to M2.** M2 asks why transformers deviate from Bayes, predicted from the architecture. Twenty-five episodes studied one
departure (a big miss earns more weight than the same squared total spread out) without knowing what share of the nets' deviation
it is. The theory lane found a second signature, cross-source interaction ("tangledness"), that tracks KL across nets. The
coordinator asked whether the two are the same phenomenon. E26 splits each net's whole deviation from Bayes by scale:
- within a source, the "pattern" part: dependence on how a source's labelled errors are spread beyond their sum of squares
  (signs, assignment to questions, and magnitude shape; the big-miss departure is one documented piece of it);
- per source, the "own, sufficient" part: errors that are functions of one source's sufficient statistics (and the context);
- across sources, "tangled": every term that involves two or more sources' data, which Bayes never has;
- "context": errors that depend only on the shared labels, identities and answer.
Each part can then be assigned to an architectural step (layer-1 pooling, the layer-2 merge, the head), which is what an
architecture-predictive account needs. The dynamics arm asks how the relation between the two signatures forms in training.
**Design** (`e26_data.py`, `e26_nets.py`, `e26_run.py`, `e26_summ.py`; checks `e26_check.py`).
- Factorisation (exact under the prior): context c = (labels, y*, identity vectors); per source s, Z_s = (v_s, rho_s, e_s) (level,
  norm of the four labelled residuals, query-reading error) and u_s = the residuals' direction on S^3, uniform and independent of
  Z_s and c. Bayes depends on source s only through (rho_s^2 = SS_s, o*_s), never on u_s, and its log-odds
  L(delta) = log p(y* + delta | x) - log p(y* | x) is exactly additive over (c, B_1, B_2, B_3), B_s = (Z_s, u_s).
- Data: 6,000 fresh bases (seed 2026102934), each with 34 variants sharing c: base; an independent all-new draw; keep B_s (3);
  keep the other two blocks (3); keep Z_s with a new u_s (3); keep all Z with new u (1); 14 more draws of all u (pattern set: 16
  draws of u given c and Z); 8 sign-and-assignment redraws (magnitudes kept). Bayes, Bayes's moment-matched Gaussian and Bayes
  reduced to 4 components (floors) on every design variant.
- Functional: D = L_net - L_Bayes at delta in {±0.15, ±0.3, ±0.6} (primary; the N5 subset ±0.3, ±0.6 also reported) and the
  score d/dy log p at y* (also reported: its decomposition is exact for the average Fisher divergence). Components (variances of
  D summed over the deltas; estimators in `e26_summ.py`'s docstring): Vtot, context Vc, own-source V_s, tangled VX (pick-freeze;
  the 2x2-cut estimate VX2 also reported), pattern total Tu (mean 16-draw variance), own pattern Vu_own (covariance estimator),
  tangled pattern Vu_cross = Tu - Vu_own, sign/assignment Tsp. Budget: context, own sufficient (sum V_s - Vu_own), own pattern,
  tangled sufficient (VX - Vu_cross), tangled pattern, as fractions of Vtot.
- KL-exact pattern cost: per base, mean over the 16 pattern draws of KL(Bayes || net) minus KL(Bayes || normalised geometric mean
  of the 16 net densities) = -log Zbar >= 0 (Jensen), times 16/15 for the finite number of draws; on a grid of 801 points over
  Bayes's mean ± 6. Pattern cost share = pooled cost / pooled mean KL. Same for the 9 sign/assignment draws (x 9/8).
- Nets (`results/e26_nets.json`): M = the 15 mixture-head nets among E22's 48 acquired depth-2 ds-16 clean nets (the 12 bases of
  the noisy nets and w16_h1_L2_mix4 s0-s2; 6,000 bases); G = the other 33 (Gaussian heads; also against the Gaussian floor); L1 =
  the 20 depth-1 ds-16 nets (none acquired); NZ = the 12 noisy nets (predictive = mixture over 8 noise draws, the same noise for
  every variant; 4,000 bases); HD = the heads cohort's base and L3 arms (E17; Gaussian and mixture heads); DY5, DY6 = E16's replays
  (every checkpoint to 10,000 steps, then every 5,000, plus the registered finals; 6,000 bases).
- Pooled ratios = sum over nets of the numerator / sum of the denominator; intervals from 1,000 bootstrap resamples of the bases
  (the same resample for every net of a set), conditional on the nets.
**Predictions** (primary functional: the six deltas).
- P137 (in acquired nets the pattern effect is own-source, not part of the tangle): set M, pooled Vu_own / Tu >= 0.7, and
  Vu_own / Tu >= 0.6 in >= 12 of the 15 nets.
- P138 (without acquisition the pattern effect is mostly cross-source): set L1, pooled Vu_cross / Tu >= 0.4.
- P139 (the pattern part is a minor share of the deviation): set M, pooled Tu / Vtot <= 0.2 and pooled pattern cost / mean KL
  <= 0.2.
- P140 (tangledness is a major share, made by the body rather than forced by the head): set M, pooled VX / Vtot >= 0.25, and the
  nets' mean VX >= 2 x the VX of Bayes reduced to 4 components.
- P141 (the pattern effect leaves the tangle when reliability weighting is acquired): in both lineages, the median of
  Vu_cross / Tu over E16's plateau-band checkpoints (s5: 1,400, 2,000, 2,250, 2,500; s6: 1,700-2,750) is >= 0.4, and its median
  over checkpoints from t_acq + 1,000 (5,250) to the end is <= 0.3.
**Also reported (descriptive, no bars):** the full budget for every set (both functionals and the N5 deltas); the A/B groups
(M's 12 bases and NZ); depth (HD: base vs L3 arms by head type); the floors; the full trajectories; the sign/assignment split;
the N5-comparable share of the net's log-odds; Spearman correlations across the 48 acquired clean nets between the pattern share,
the tangled share, KL, the pattern cost share, E15's large-range rom and the own share of the pattern.
**What the outcomes would mean.**
- P137 and P140 hold: the big-miss departure is an own-source error at the token-to-source scale and the tangledness is a
  separate, larger error at the source-to-root scale; the two signatures are different phenomena (the theory note's conjecture
  C2 would then need two scales). P137 fails: the departure is substantially entangled with the other sources' data, i.e. partly
  the same phenomenon.
- P138 and P141 hold: the big-miss sensitivity starts as part of a tangled, non-specific solution and separates into an own-source
  error when the net acquires source-specific reliability weighting (a statement about how these nets learn).
- P139 holds: the big-miss family is a small share of why the nets deviate; most of the deviation lies elsewhere (tangled or
  own-sufficient), and the M2 work should move there. P139 fails: the pattern part is a major share and the departure line stays
  central.
- P140 fails: tangledness is minor in acquired nets, or no larger than what a 4-component head forces.
- The two functionals weight errors differently (the log-odds metric emphasises the posterior's shape; KL does not), so shares
  are stated per metric; only the pattern cost is a share of KL.


## Amendment 35 (2026-10-29 21:30Z): E27, which sign and assignment information do the nets use?

Written before the E27 data file exists and before any E27 quantity of a registered net is computed. Known beforehand: E26's
registered results and its post-hoc analyses (`e26_post.py`), before E26's review:
- in the 15 mixture-head nets the pattern part costs 31% of the KL; signs and assignment (E26's 9-draw set: random signs and a
  random assignment to the labelled questions, independently per source, magnitudes kept) cost 20%;
- in the 6 B bases these costs are 39% and 29% of KL, in the 6 A bases 16% and 8%; they account for 74% and 60% of B's extra KL;
- half of B's pattern effect is cross-source (A: 31%), and B's extra tangledness over A is entirely pattern-involving.
Plumbing (disclosed): `e27_data.py` and `e27_run.py` on a check seed (777, 500 bases) with a random-init net: Bayes and every SS are
unchanged under all four families (max change 7e-15); joint flips keep the cross-source sign agreement on every question; joint
permutations keep which residuals share a question; the random net's four costs are equal.
**Link to M2.** E26 found that the largest within-source error, and the main difference between the two solution types, is a
dependence on the signs and question assignment of a source's labelled errors, which Bayes ignores. An architecture-predictive
account needs the computation behind it. Three candidates:
- (consensus) the nets judge a source partly by how its labelled readings agree with the other sources' readings on the same
  question. For the unlabelled query question this is what Bayes does (the query readings' joint marginal enters the posterior over
  the noise levels). Every token is processed by the same weights, so the computation may leak into the labelled tokens, where
  the label makes it redundant. This would be an inductive bias of weight sharing, predictable from the architecture;
- (bias) a source's mean labelled error is read as a bias of its reading;
- (label) a residual's representation depends on the label or question identity it sits with.
**Design** (`e27_data.py`, `e27_run.py`, `e27_summ.py`). 6,000 fresh bases (seed 2026102935). Four families of 8 redraws each,
keeping the context, every source's level, query error and residual magnitudes (so Bayes is unchanged):
- flip_ind: independent signs per source and question;
- flip_joint: one sign per question shared by the three sources, which keeps how every labelled reading sits relative to the
  others up to a common sign;
- perm_ind: each source's residuals permuted across its labelled questions independently;
- perm_joint: one permutation for all three sources, which keeps which residuals share a question.
Per net and family: KL-exact cost (as E26, 9 samples, x 9/8), its share of KL, and the 9-sample variance of the log-odds; the
block-1 attention mass from each labelled token to the other two sources' tokens of the same question (xq; uniform = 2/15).
Nets: E26's sets M and G (the 48 acquired clean nets); groups A and B = E26's bases. Pooled ratios = sums over nets; intervals
from 1,000 bootstrap resamples of the bases, conditional on the nets.
Predictions under the three candidates:
- consensus: flip_joint much cheaper than flip_ind, and perm_joint much cheaper than perm_ind;
- bias: flip_joint about as costly as flip_ind, and both permutation costs near zero (the mean error does not change);
- label: flip_joint about as costly as flip_ind, and perm_joint about as costly as perm_ind, both above zero.
**Predictions** (registered for the consensus account, which E26's cross-source share in B favours):
- P142a (signs): in the 6 B bases, pooled cost(flip_joint) <= 0.5 x cost(flip_ind).
- P142b (assignment): in the 6 B bases, pooled cost(perm_joint) <= 0.5 x cost(perm_ind); testable only if cost(perm_ind) >= 5%
  of their KL (otherwise the nets do not use the assignment, and P142b is reported as untestable).
- P143 (consensus separates the groups): of the B - A difference in cost(flip_ind) + cost(perm_ind), the agreement-breaking part
  [cost(flip_ind) - cost(flip_joint)] + [cost(perm_ind) - cost(perm_joint)] is >= 50%.
**Also reported (descriptive):** every family's cost by group (A, B, all of M, G) and per net; the variance-based version; the
Spearman correlation across the 48 nets between xq and the agreement-breaking cost share.
**What the outcomes would mean.** P142a and P142b hold: the nets use cross-source agreement on labelled questions, an inductive
bias the architecture predicts from weight sharing between labelled and query tokens (to be tested next by changing that sharing).
P142a fails with near-zero permutation costs: a bias reading of the mean error. P142a and P142b fail with equal costs: label or
question leakage. P143 fails: what separates B from A is not the agreement-based part.


## Amendment 36 (2026-10-29 22:01Z): E28, is the nets' sign dependence a reading of each source's mean error?

Written before the E28 data file exists and before any E28 quantity of a registered net is computed. Known beforehand: E26 and
E27 in full, with their reviews:
- the error-pattern dependence costs about 29% of KL in the mixture-head nets (E26); it is mostly sign dependence (E27: sign flips
  cost 6.8% of KL in the A bases and 24.6% in the B bases, uncorrected);
- shared and independent sign flips cost the same, and question assignment matters little (0.3-0.4%);
- in the B bases half of the pattern effect is cross-source (E26).
Plumbing (disclosed): `e28_data.py`, `e28_run.py` and `e28_summ.py` on a check seed (777, 500 bases) with a random-init net and
copies of its output. Bayes and every SS are unchanged under all families (max change 4e-15). Negation keeps every within-source
product of residuals; shared negation keeps every cross-source product; each negation pair negates only its source.
**Link to M2.** A dependence on the signs of a source's errors, which Bayes ignores, is the largest measured component of group
B's extra distance from Bayes. One architectural account would be a first-moment leak:
- the source's summary pools linear embeddings of its residuals, so it carries their mean (odd in the signs) alongside the
  second-moment information Bayes needs;
- what reads the summary, or the source's reading on the same token, may not cancel the mean.
This would be predictable from the architecture (pooling of linear token features). The alternative is a reading of the
within-source sign pattern (even under negating all of a source's errors), such as treating same-signed errors as less noisy,
which a sample variance around the mean would do.
**Design** (`e28_data.py`, `e28_run.py`, `e28_summ.py`). 6,000 fresh bases (seed 2026102936). Families of 8 draws (plus the base),
keeping the context, levels, query errors and magnitudes:
- flip_ind: independent sign flips (as E27);
- negate_ind: each source's whole residual vector times its own random sign; this reverses its mean error and keeps its sign
  pattern;
- negate_joint: all three sources' vectors times one common sign, which also keeps every cross-source product.
Three negation pairs (the base with one source negated). Per net (the 48 acquired clean nets of E26's sets M and G):
- KL-exact costs (uncorrected) per family;
- for the base and each negation pair, the predictive mean m and the effective weights w_s = dm/do*_s (autograd).
Measures:
- odd share = cost(negate_ind) / cost(flip_ind), pooled;
- mean-shift regression: dm_s = m(base) - m(base with s negated) on X1 = 2 rbar_s w_s (own mean error times the source's
  weight in the net) and X2 = 2 rbar_s (w_1 + w_2 + w_3) / 3 (mean error pooled over the sources), OLS without intercept, R^2
  against sum dm^2. A shift of source s's reading by -c rbar_s gives dm_s = -c X1;
- the weight change dw_s on negating s, against 2 rbar_s (descriptive).
**Predictions** (for the mean-error reading; groups as in E26).
- P144 (mostly odd): in the B bases, pooled cost(negate_ind) >= 0.5 x cost(flip_ind).
- P145 (it acts on the mean): in the B bases, the pooled regression of dm on (X1, X2) has R^2 >= 0.3.
- P146 (a consistent direction): the own coefficient has the same sign in >= 5 of the 6 B bases, with absolute value >= 0.1 in
  those. No direction is registered: correction (negative) and anti-correction (positive) are both possible under a leak.
**Also reported (descriptive):** every group's costs (A, B, all M, G), cost(negate_joint) / cost(negate_ind), the own-only
regression, the weight-change regression, per-net values.
**What the outcomes would mean.**
- P144-P146 hold: the nets read a source's mean labelled error and shift their prediction by it, consistently in one direction.
  This is a first-moment leak that the architecture predicts; the next tests would change the pooling, e.g. sign-symmetric
  token features.
- P144 holds and P145 fails: the odd part acts through reliability (weights), not the mean.
- P144 fails: the dependence is mostly on the sign pattern within a source (even), e.g. a variance around the sample mean.
- negate_joint much cheaper than negate_ind: cross-source products of signs matter (part of the B nets' cross-source pattern
  effect).


## Amendment 37 (2026-10-29 22:22Z): E29, is the sign dependence tied to the random identity vectors?

Written before the E29 data file exists and before any E29 quantity of a registered net is computed. Known beforehand: E26-E28
in full with their reviews. In particular:
- in the B bases most of the sign dependence goes with a source's overall error direction (E28 cost ratio 0.80), with little
  dependence on question assignment (E27) or on products of two sources' signs (E28);
- the negation-induced mean shift is not proportional to the mean error times the weight, and fitted predictors on the source's
  own error data explain 10-28% of it on held-out episodes (E28, post hoc).
Plumbing (disclosed): `e29_data.py` and `e29_run.py` on a check seed (777, 500 bases) with a random-init net. Bayes is
unchanged across all variants (4e-15). Each identity variant changes only the intended vectors. The random net's negation
shift keeps 90% of its direction under a new source identity vector.
**Link to M2.** The tokens carry random per-episode identity vectors, the only way these nets can tell sources and questions
apart (no positional encoding). If the sign dependence is entangled with those vectors, it is an interference between the
binding code and the content inside the same token representation. That would be an inductive bias of this architecture and
input format: odd in the errors, independent of the questions, and not predictable from the errors alone, as E27-E28 found.
**Design** (`e29_data.py`, `e29_run.py`, `e29_summ.py`). 6,000 fresh bases (seed 2026102937), 26 variants:
- the base;
- each source negated;
- each source's identity vector redrawn, with and without negating that source (the same new vector for both);
- 8 redraws of all three source identity vectors;
- 8 redraws of all five question identity vectors.
Bayes is unchanged across all of them. Per net (the 48 acquired clean nets of E26's sets M and G):
- the predictive mean of the first ten variants;
- the KL-exact cost (uncorrected) of each identity family.
Measures:
- stable fraction phi = sum dm dm' / sum dm^2, with dm_s = m(base) - m(base, s negated) and dm'_s the same under s's
  redrawn identity vector. If the shift is h(errors, ...) + g(errors, identity vector of s), with g averaging to zero over
  identity vectors, phi estimates the share of its variance that is not tied to that vector;
- the source-ID and question-ID costs as shares of the mean KL.
Pooled over nets; intervals from 1,000 bootstrap resamples of the bases, conditional on the nets.
**Predictions.**
- P147 (tied to the identity vector): in the B bases, pooled phi <= 0.5.
- P148 (B nets depend more on the random source identities): the source-ID cost share in the B bases is >= 2x that in the A
  bases.
- P149 (question identities matter little, as E27's permutations suggested): the question-ID cost share is <= 0.05 of KL in
  both groups.
**Also reported (descriptive):** phi, correlation and costs for all 15 mixture-head nets and the 33 Gaussian-head nets;
per-net values.
**What the outcomes would mean.**
- P147 holds: most of the odd shift changes with the source's random identity vector. The sign dependence is then an
  interference between identity and content, and the next tests change the identity code (its size, or a fixed code). P147
  fails: the shift is a stable function of the episode that does not depend on the identity vector, and the fitted predictors
  missed it.
- P148 holds: group B's extra error is tied to the random identities, which fits its larger cross-source attention (E27).


## Amendment 38 (2026-10-29 22:51Z): E30, do input formats with built-in invariances bring retrained nets closer to Bayes? (GPU, approved)

Written before any E30 training. Known beforehand: E26-E29 with their reviews. In the mixture-head nets:
- geometric averaging over sign flips of the labelled errors removes about 17% of the KL (E27, E28);
- averaging over redrawn source codes removes about 21-27% (E29, A/B bases 21.5%/23%, all mixture-head nets 27%);
- the two overlap, and their joint reduction was not measured.
The coordinator approved E30 (on-demand L4, at most $3) with three design points, all taken up below: a representation control
for the magnitude format, the binding caveat for fixed codes, and predicted reductions from the measured costs, plus A/B types
for the control seeds.
Plumbing (disclosed):
- `train_fmt.py` and `e30_fmt.py`: unit checks show the ctrl tokens are byte-identical to the original builder; mag changes only
  the labelled readings; sgnmag adds one feature; fix uses constant codes;
- 5,000-step smoke trainings of every arm with seed 0 (not converged; not part of the results) gave test gaps ctrl 0.078, mag
  0.021, sgnmag 0.047, fix 0.031. At that stage the sgnmag net's sign-flip KL reduction was 0.2% of its KL against ctrl's 16.7%.
  So the formats change early learning speed, and a net given the magnitude may drop the signs by itself;
- `e30_eval.py` and `e30_summ.py` ran on those smoke nets.
**Link to M2.** If a large part of the deviation is a failure to learn the posterior's invariances, an architecture or input
format that has them built in should lose that part. This would predict the departure from the architecture in a quantitative
way: the KL reduction should match the invariance costs measured on ordinary nets. If the reduction does not appear, or appears
only through easier inputs, the invariance account of these costs is wrong or incomplete.
**Design** (`e30_fmt.py`, `train_fmt.py`, `gpu/gpu_e30.sh`, `gpu/launch_e30.sh`, `e30_eval.py`, `e30_summ.py`).
- The tiny-pfn recipe (2 layers, width 32, 2 heads, mixture head K = 4, AdamW, the 40k/80k convergence rule and 10k cooldown,
  `pfn2.py` unchanged) in four input-format arms, seeds 0-11 each. With the same seed, every arm has the same initialisation and
  the same episodes. The arms:
  - ctrl: the original format;
  - mag: labelled tokens carry y_q + |o_q - y_q| instead of o_q. The signs of the labelled errors are removed, so the net is
    exactly sign-invariant; Bayes is unchanged;
  - sgnmag (representation control, the coordinator's point 1): the original signed readings plus one extra feature, |o_q - y_q|,
    on labelled tokens. The magnitude is as directly available as in mag, but the signs stay visible;
  - fix: fixed identity codes in every episode. This also makes binding trivial (the coordinator's point 2); the fix arm is
    therefore compared with the ctrl nets after test-time averaging over codes.
- One on-demand L4 VM, up to 8 trainings at once, interleaved by seed. The job's own deadline: no new training after 8,400 s,
  everything stopped at 9,900 s, so the VM stays under about $2.5. Every VM is logged in `GPU-LOG.md`.
- Measures on CPU (`e30_eval.py`):
  - test KL (gap on the 16,000 tiny-pfn test episodes in each net's own format);
  - E15's big-miss rom on the E12 pairs, and the A/B type from the sign of the middle-range rom;
  - on E27's data, the sign-flip KL reduction (ctrl, sgnmag, fix);
  - on E29's data, the source-code KL reduction (ctrl, mag, sgnmag).
- Acquired = gap < 0.10. K_arm is the mean KL over an arm's acquired nets; c_sign and c_code are the ctrl nets' pooled reduction
  shares. Intervals come from 2,000 bootstrap resamples of seeds, paired across arms.
**Predicted reductions** (the coordinator's point 3). From E26-E29 we expect c_sign of about 0.17 and c_code of about 0.21-0.27 in
the new ctrl nets. If the measured costs carry over to retraining:
- K_mag would be about 0.83 x K_ctrl;
- K_fix would be at most about 0.79 x K_ctrl, or lower if easy binding adds a gain of its own.
**Predictions.**
- P150 (the sign cost carries over): 1 - K_mag/K_ctrl lies in [0.5, 1.5] x c_sign, with c_sign measured on the new ctrl nets.
- P151 (easy access to the magnitude is enough): K_sgnmag <= 1.1 x K_mag, and the sgnmag nets' sign-flip reduction share is
  <= 0.25 x the ctrl nets'. (Informed by the smoke run; see above.)
- P152 (the code cost carries over, at least half): 1 - K_fix/K_ctrl >= 0.5 x c_code.
- P153 (B-type seeds gain more from removing the signs): among seeds acquired in ctrl and mag, the mean paired reduction
  K_ctrl - K_mag is larger for seeds whose ctrl net is B type than for A-type seeds. Testable only with >= 2 of each.
**Also reported (descriptive):**
- K_arm, medians and A/B counts per arm; the big-miss rom per arm;
- K_fix against the code-averaged ctrl KL, K_ctrl x (1 - c_code). Clearly below means easy binding gains beyond the invariance;
  about equal means the invariance is the gain;
- K_mag against the sign-averaged ctrl KL, and K_sgnmag against K_ctrl.
**What the outcomes would mean.**
- P150 holds: the sign-invariance failure measured on trained nets predicts what an invariant format gains.
- P151 holds: the gain comes from easy access to the magnitude, not from imposing the symmetry. The ordinary nets' sign dependence
  would then be the cost of computing an even function of a signed input with these layers.
- P150 fails high: the format helps beyond the measured cost (e.g. learning speed). Fails low: the cost does not carry over.
- P152 holds and K_fix is near the code-averaged ctrl: the code cost carries over. K_fix well below it: easy binding adds more.


## Amendment 39 (2026-10-29 22:54Z): E31, when do the two lineages lose their sign dependence and their cross-source attention?

Written before any E31 quantity is computed (CPU, run in parallel with E30's GPU job). Known beforehand: E26-E29 with their
reviews; E26's trajectories of the total pattern cost over a subset of E16's checkpoints. The A lineage's pattern cost fell from
0.0079 to 0.0031 nats between 15,000 and 20,000 steps, rose to 0.0074 at 30,000 and fell again by 40,000. The B lineage's fell
slowly. E27's marker xq on the final nets is B base_mix4_s5 0.046 and A base_mix4_s6 0.002.
**Link to M2 (how they learn).** The A/B split is the main difference in these nets' distance from Bayes (E26). If the A lineage
loses its sign dependence in a late transition, and the cross-source same-question attention falls at the same time, the two
group-defining properties are learned or unlearned together. That would point to one mechanism behind the B type's error.
**Design** (`e31_run.py`, `e31_summ.py`). Every checkpoint of E16's two replays (123 each), on the first 3,000 bases of E28's data
(registered, Bayes-invariant):
- the KL-exact cost (uncorrected) of independent sign flips and of whole-vector negation (9 samples each);
- the mean KL;
- the block-1 marker xq on the base variant.
**Predictions** (window medians: W1 = 10,000-15,000 steps, W2 = 20,000-30,000 steps).
- P154: A lineage, median sign-flip cost in W2 <= 0.5 x W1; B lineage, >= 0.7 x.
- P155: A lineage, median xq in W2 <= 0.5 x W1; B lineage, >= 0.7 x.
**Also reported (descriptive):** full trajectories of KL, both costs, their shares of KL, the odd share and xq.
**What the outcomes would mean.**
- P154 and P155 both hold: the A lineage loses the sign dependence and the cross-source attention in the same window, a joint
  late transition.
- P154 holds and P155 fails: the sign dependence is lost without a change in that attention. The marker and the error are then
  separate.


## Amendment 40 (2026-10-29 23:27Z): E32, does block 1's cross-source same-question attention carry the sign dependence?

Written before any E32 quantity is computed (CPU, in parallel with E30's GPU job). Known beforehand: E26-E31; E31's
result (not yet reviewed):
- in the A lineage, the cross-source same-question attention (xq) and the sign dependence fall together between 12,000 and
  20,000 steps;
- the B lineage keeps both;
- E27's final-net xq: A bases 0.002-0.004, B bases 0.046-0.081.
Plumbing (disclosed): on a random-init net, the re-implemented forward without the patch equals the model's output, and the patch
changes it; the natural xq there is 0.142 (uniform 2/15 = 0.133).
**Link to M2.** E31 suggests a mechanism for the B type's largest extra error. A labelled token pulls in the other sources'
readings on the same question, whose residuals are odd in the signs, and these enter the source summaries. If switching that
attention off removes most of the sign dependence, a specific attention pattern carries the B type's extra error. The account
then names the architectural route (unrestricted attention among labelled tokens of different sources).
**Design** (`e32_run.py`, `e32_summ.py`).
- The 48 acquired clean nets of E26's sets M and G, on the first 3,000 bases of E28's data.
- Patch: in block 1, for every labelled token (s, q) and every head, the attention weights on (t, q), t != s, are set to zero and
  the row is renormalised; everything else is unchanged.
- Measured unpatched and patched: the sign-flip and negation costs (KL-exact, uncorrected, 9 samples), the base KL and the
  natural xq.
- Groups: A and B bases, all 15 mixture-head nets, the 33 Gaussian-head nets.
- Caveat registered now: the patch is off-distribution for later layers (E22), so a KL rise can be disruption rather than
  evidence against the account. The removed mass is small in B (5-8% of each labelled token's attention) and very small in A.
**Predictions.**
- P156 (that attention carries the sign dependence in B): in the B bases, the pooled sign-flip cost under the patch is <= 0.5 x
  the unpatched.
- P157 (removing it brings the B nets closer to Bayes): in the B bases, the patched base KL is below the unpatched in >= 4 of 6.
**Also reported (descriptive):** the A bases as a control (little attention to remove); all mixture-head and Gaussian-head nets;
negation-cost ratios; per-net values.
**What the outcomes would mean.**
- P156 and P157 hold: the B type's extra sign dependence runs through cross-source attention among labelled tokens. Restricting
  that attention architecturally would be the next test.
- P156 fails: the attention is a marker of a B-type solution whose sign dependence is carried elsewhere.


## Amendment 41 (2026-10-29 23:40Z): E33, which block-1 outputs carry the sign-odd dependence to block 2?

Written before any E33 quantity of a registered net is computed (CPU, in parallel with E30's GPU job). Known beforehand: E26-E32
with their reviews. E32 found that switching off the cross-source same-question attention leaves the sign dependence nearly
unchanged.
Plumbing (disclosed), on the random-init net with E28's data:
- the re-implemented split forward (block 1, replacement, block 2, head) equals the model when nothing is replaced;
- replacing all tokens gives identical outputs across the eight negation patterns;
- the 'all' arm's measured cost there is 0.0057, not 0: the grid truncation baseline for that net's wide densities. Removal
  shares are therefore measured relative to the 'all' arm.
**Link to M2.** E24-E25 located the big-miss contrast in each source's own pooling (its summary token, the query-question token,
pooling its labelled tokens in block 1). If the same summary tokens also carry most of the sign-odd dependence, both departures
come from one computation, pooling linear token features. Its even part caps per-error evidence; its odd part leaks the signs.
That would be one architectural account for two error components.
**Design** (`e33_run.py`, `e33_summ.py`). The 48 acquired clean nets of E26's sets M and G, on the first 3,000 bases of E28's
data.
- For each base, block 1's output (the residual stream after block 1) is computed for its eight negation patterns. These negate
  any subset of the three sources' labelled-residual vectors and all have the same Bayes posterior; E28's negation family lies
  among them.
- An arm replaces that output at a set of tokens by its average over the eight patterns, in every pattern, then runs block 2 and
  the head unchanged. Arms:
  - none;
  - summ: the three summary tokens;
  - lab: the twelve labelled tokens;
  - query: the query token;
  - all: every token (sets the zero).
- Measures per arm: the negation cost (uncorrected KL reduction from geometric averaging over E28's 9 negation-family samples)
  and the base KL.
- removed(arm) = (cost_none - cost_arm) / (cost_none - cost_all), pooled over a group's nets.
- The averaged activations include cross-source terms, and block 2 is nonlinear. Removal shares are evidence about those
  activations, not an additive attribution, and the replacement is off-distribution for block 2.
**Predictions.**
- P158 (the summary tokens carry most of it): in the B bases, removed(summ) >= 0.5.
- P159 (the odd part is error): in the B bases, the base KL under 'all' is <= 0.9 x the base KL under 'none'.
**Also reported (descriptive):** removed(lab), removed(query) and KL ratios for every arm; the A bases, all mixture-head nets and
the Gaussian-head nets; per-net values.
**What the outcomes would mean.**
- P158 holds: the same summary computation carries both the big-miss contrast and most of the sign dependence. One pooling step
  accounts for two error components.
- P158 fails, with lab or query large: the signs reach block 2 directly from the labelled tokens or through the query token,
  bypassing the source summaries.
- P159 holds: averaging the sign-odd part away inside the net brings it closer to Bayes, as the output-level cost suggests.


## Amendment 42 (2026-10-29 23:53Z): E34, how much of the sign dependence travels by block 2's query read of the labelled tokens? (path patching)

Written before any E34 quantity of a registered net is computed (CPU, in parallel with E30). Known beforehand: E26-E33 with their
reviews. In particular, E33 found post hoc that averaging the labelled tokens' block-1 states over sign patterns raises the A bases'
KL by at most 4% and the B bases' by 2.7-6.1x. The head reads only the query token, so the B nets' output depends on those states
through block 2's query attention.
Design (the coordinator's suggestion, replacing an earlier draft that zeroed this attention): path patching with real
counterfactual states. Every patched state is then a real state for another sign pattern.
Plumbing (disclosed), on the random-init net with E28's data: the re-implemented block-2 query computation equals the model for
every orbit member, and the path hybrid with the base member equals the base.
**Link to M2.** The B type's extra error is mostly sign dependence (E26-E28). If one path, block 2 reading the labelled tokens'
states, carries most of it in B and little in A, then the two solution types differ in where the merge reads from: B reads
per-question signed states, A reads source summaries. That would be a route-level account of the A/B difference and of B's extra
error, predictable from which attention pattern a net learns.
**Design** (`e34_run.py`, `e34_summ.py`). The 48 acquired clean nets of E26's sets M and G; the first 3,000 bases of E28's data.
For each base, the eight negation patterns of E33 give real block-1 states. Two hybrids per pattern m, at block 2's query row:
- path: the labelled tokens' keys and values come from m; everything else (the query's own state and query vector, the other
  tokens' keys and values) from the base. Only this path sees m's signs;
- rev: everything from m except the labelled tokens' keys and values, which come from the base.
Measures:
- on the three single-source negations, share_path = Σ dm_path dm_full / Σ dm_full² and share_rev = Σ dm_rev dm_full / Σ dm_full²,
  with dm_full = m(negated) − m(base), dm_path = m(path hybrid) − m(base), dm_rev = m(negated) − m(rev hybrid);
- the negation cost (KL-exact, uncorrected, E28's 9-sample family) with every sample replaced by its path hybrid (path-only) or its
  rev hybrid (all-but-path), as a share of the full cost;
- the KL of the hybrid families (side measure);
- block 2's query attention mass on the labelled and summary tokens.
**Predictions.**
- P160 (B: this path carries most of it): in the B bases, pooled share_path >= 0.5 and path-only cost >= 0.5 x the full cost.
- P161 (A: it carries little): in the A bases, pooled share_path <= 0.3, and below the B bases'.
**Also reported (descriptive):** share_rev and the all-but-path cost; hybrid KLs; attention masses; all 15 mixture-head and 33
Gaussian-head nets; per-net values. Shares from nonlinear processing need not add to 1.
**What the outcomes would mean.**
- P160 and P161 hold: B nets' sign dependence travels mainly by block 2 reading labelled tokens directly, a route the A nets barely
  use. This identifies a route difference between the solution types.
- P160 fails: B's sign dependence comes mostly through the summary tokens or the query token's own block-1 state, despite B's
  sensitivity to the labelled tokens' states in E33.


## Amendment 43 (2026-10-30 00:04Z): E35, does block 2's query attention to the labelled tokens fall in the A lineage together with its sign dependence?

Written before any E35 quantity is computed (CPU, small; in parallel with E30). Known beforehand: E31 (in the A replay, block 1's
cross-source same-question attention and the sign dependence fall together between 12,000 and 20,000 steps) and E34 (block 2's
query attention to the labelled tokens, per base: A 0.06-0.28, B 0.39-0.45; the A replay's base 0.256, the B replay's 0.387).
**Link to M2 (how they learn).** E34 found a block-2 routing difference between the types. If the A lineage reduces its block-2
reading of the labelled tokens in the same late window in which it loses its sign dependence and its block-1 cross-source
attention, the A type's refinement is one coordinated change of routing.
**Design** (`e35_run.py`, `e35_summ.py`). Every checkpoint of both E16 replays, on the base episodes of the first 3,000 bases of
E28's data: the query token's block-2 attention mass on the twelve labelled tokens and on the three summary tokens.
**Prediction.**
- P162: A lineage, median labelled mass in 20,000-30,000 steps <= 0.7 x its median in 10,000-15,000; B lineage, >= 0.9 x.
The final checkpoints' masses are a consistency check against E34's bases, not a prediction.
**What the outcomes would mean.** P162 holds: the A lineage's late refinement includes moving block 2's reading from the labelled
tokens toward the summaries. P162 fails: the block-2 routing difference is set at another time than the sign dependence's
decline.


## Amendment 44 (2026-10-30 00:10Z): E36, which routes carry the single-source negation shift? (path patching of block-1 routes)

Written before any E36 quantity of a registered net is computed (CPU, in parallel with E30). Known beforehand: E26-E35 with
their reviews. E34: block 2's query read of the labelled tokens reproduces, as a projection, 0.31 of the B bases' single-source
negation shift and 0.12 of the A bases'. Plumbing (disclosed): on the random-init net the split forward (block-1 attention output,
block-1 MLP, block-2 query row, head) reproduces the model's predictive mean.
**Link to M2.** E34 left most of B's sign shift unattributed. The remaining candidate routes are built in block 1: each source's
summary token, which pools its labelled tokens (where E24-E25 located the big-miss contrast), and the query token, which also
attends to labelled tokens in block 1. Finding which carries the shift completes a route-level account of the sign dependence and
of the A/B difference.
**Design** (`e36_run.py`, `e36_summ.py`). The 48 acquired clean nets of E26's sets M and G; the first 3,000 bases of E28's data.
For each base and each single-source negation s (real states), the predictive mean of hybrids that take one computation from the
negated run and everything else from the base run:
- summ_own: block 1's attention output at source s's summary token;
- summ_all: at all three summary tokens;
- query1: at the query token;
- lab2: E34's path, block 2's query read of the labelled tokens.
Summary and query tokens have identical embeddings in both runs, so only their attention outputs differ; the block-1 MLPs and
block 2 are recomputed. Shares are projections, share_r = Σ dm_r dm_full / Σ dm_full², pooled over a group's nets. The squared size
of each hybrid's shift relative to the full one is also reported.
**Predictions.**
- P164 (the source's own summary carries a substantial part in B): in the B bases, share(summ_own) >= 0.3.
- P165 (A routes through summaries): in the A bases, share(summ_own) > share(lab2).
**Also reported (descriptive):** all routes and squared sizes for A, B, all mixture-head and Gaussian-head nets; per-net shares.
Projections of nonlinear effects need not add to 1.
**What the outcomes would mean.**
- P164 holds: the sign dependence in B travels partly through the source summaries (the pooling site of E24-E25) and partly through
  block 2's direct read (E34).
- P165 holds: in A the summary route dominates, consistent with block 2 reading summaries (E34).
- If the shares sum well below 1, the shift arises from nonlinear combinations of routes rather than any single one.


## Amendment 45 (2026-10-30 02:05Z): E37, when is the solution chosen? A format-switch test of the selection hypothesis (GPU)

Written by the fourth agent before any E37 training and before any E37 quantity is computed. Known beforehand: E1-E36 and E30
with their reviews. In particular:
- E30 (natural-length recipe): the sign-free format mag reaches 0.36x the ctrl KL on paired seeds (0.47 on arm means); every seed
  improves (B-type ctrl seeds 1.8-5.4x, A-type 1.3-2.3x); 11 of 12 mag nets are A type against 4 of 11 ctrl nets; the gain is
  about 3.2 times the sign cost measured on trained ctrl nets. The per-seed values are in Update 46: ctrl seed 3 stalled (0.142);
  mag seed 3 is the only B-type mag net (0.022). E30's 5,000-step smoke runs: test gap ctrl 0.078, mag 0.021.
- E16, E26, E31, E35 (one lineage of each type): reliability weighting, the big-miss departure and the sign dependence appear in
  one 250-step interval at 2,500-3,000 steps. Afterwards the B replay's block-2 query attention to the labelled tokens rises (0.09
  at 3,000 steps, 0.24 at 10,000, 0.38 at the end) and the A replay's falls modestly (0.31 to 0.26); at 1,000 steps they are
  0.165 (B) and 0.348 (A). The A replay's block-1 cross-source attention and sign dependence fall at 12,000-20,000 steps.
- Observed while preparing this amendment, from existing result files only: E30's GPU-trained ctrl nets reproduce the original
  CPU-trained bases of the same seeds (w32_h2_L2_mix4 s0-s7, the same initialisation and episode stream) to 3-4 digits in test
  gap and middle-range rom (e.g. s1 0.0079 / 0.0079, s4 0.0148 / 0.0148). This training is numerically stable rather than
  chaotic, and a seed's A/B type is reproducible.
Plumbing (disclosed; no registered training):
- `train_sw.py` on CPU, ctrl, seed 5, 1,000 steps: its step-0 and step-1,000 weights equal those of E16's replay dyn_base_s5 bit
  for bit, so the script reproduces the recipe's initialisation, episodes, loss, optimiser and lr;
- a switch arm (ctrl to mag after 500 steps) equals the pure ctrl arm at step 500 bit for bit and differs afterwards;
- `e37_eval.py` on that step-1,000 checkpoint (E16's, already measured in E31 and E35) gives block-2 labelled mass 0.165, summary
  mass 0.819, xq 0.069, sign-flip cost 0.0024 nats and KL 0.224, matching E31/E35's 0.165, 0.819, 0.069, 0.0024 and 0.225;
- `e37_traj.py` ran on the smoke nets; `e37_summ.py`, `e38_summ.py` and `e39_summ.py` ran on synthetic inputs and recovered the
  planted values.
**Link to M2 (how they learn; why they cannot learn better).** E30 found that removing the error signs from the input changes the
attained KL by 2-3x, about three times the sign dependence measured on trained nets, and turns nearly every net into the A type.
One explanation is selection: the input available when reliability weighting forms (2,500-3,000 steps) decides which solution
training settles in, and that solution persists. Part of these nets' distance from Bayes would then be a property of the
training path, not of what the architecture can represent: the same architecture, given the same sign-free inputs later, would
not reach the better solution. This is a critical period in the sense of Achille, Rovere and Soatto (2019, "Critical learning
periods in deep networks"): a deficit removed early allows full recovery, one removed late does not. The alternatives:
- the signs cost continuously while present, so a switch at any time rescues, given the remaining budget;
- the sign-free format only learns faster, so a late switch does as well as a from-scratch net with the same remaining budget.
**Design** (`train_sw.py`, `gpu/gpu_jobs.sh`, `gpu/launch_jobs.sh`, `gpu/retry_jobs.sh`, `gpu/make_jobs.py` and its job files,
`e37_eval.py`, `e37_traj.py`, `e37_summ.py`; `e30_fmt.py`, `e30_eval.py`, `pfn2.py` and `world.py` unchanged).
- The tiny-pfn recipe (2 layers, width 32, 2 heads, mixture head K = 4; pfn2.py's model, loss, AdamW, clipping and data), seeds
  0-11, with a fixed length for every arm: warmup 1,000, constant lr to 80,000 steps, linear cooldown to 90,000. This is the
  recipe's extended length; the convergence rule is not used, so no arm's length depends on its switch. With the same seed, every
  arm has the same initialisation and the same episodes. Arms:
  - ctrl, mag: one format throughout (the endpoints);
  - c2m2k, c2m3k, c2m6k, c2m10k, c2m30k: ctrl for the first K = 2,000, 3,000, 6,000, 10,000 or 30,000 steps, then mag;
  - m2c3k, m2c10k, m2c30k: mag for the first K steps, then ctrl (the reverse switch);
  - mag60: mag throughout, constant lr to 50,000 and cooldown to 60,000 steps, the budget c2m30k has after its switch (the
    learning-speed control).
- Model selection: the best validation NLL (pfn2's 4,096 validation episodes, every 1,000 steps) in the final format, among the
  evaluations after the switch. Checkpoints: step 0; 500, 1,000, ..., 90,000 (25 steps); K + 250, 500, 1,000, 2,000 and 5,000.
- GPU: on-demand L4 (g2-standard-8), up to 8 trainings at once. E37 and E39 seeds 0-5 run on one VM and seeds 6-11 on another (72
  trainings each, ordered by seed). Own deadline: no new training after 15,600 s, hard stop at 17,400 s; runner timeout 18,600 s.
  Every VM is logged in `GPU-LOG.md`, and the instance list is checked after deletion. Estimated cost of the two VMs: about $6.5.
- Measures (CPU, `e37_eval.py`, nice 19):
  - K = test KL (gap on tiny-pfn's 16,000 test episodes) in the final format;
  - E15's rom (large and middle range) on the E12 pairs; type A if the middle-range rom is < 0, B if > 0;
  - lab = block 2's query-row attention mass on the 12 labelled tokens (E34/E35) and xq (E27), on the base episodes of E28's first
    3,000 bases, in the final format;
  - the sign-flip KL reduction (E27's data) for nets that end in ctrl.
- Trajectories (`e37_traj.py`, descriptive): at every checkpoint, test KL on 4,000 test episodes in both formats, lab and xq on
  1,000 bases in the format in force, and rom at 3,000, 6,000, 10,000, 20,000, 30,000, 40,000, 60,000, 90,000 and K + 5,000.
- Quantities: R(a) = sum_s (K_ctrl,s - K_a,s) / sum_s (K_ctrl,s - K_mag,s), over seeds acquired (gap < 0.10) in ctrl, mag and a: how
  mag-like arm a ends (1 = as mag from scratch, 0 = as ctrl). Rlab(a) is the same for lab; it is testable if the mean of
  lab_ctrl - lab_mag over those seeds is at least 0.05 in absolute value. Intervals: 2,000 bootstrap resamples of seeds, paired.
**Predictions** (the selection hypothesis).
- P166 (an early switch rescues): R(c2m2k) >= 0.8.
- P167 (a late switch does not, and the remaining budget is not the reason): R(c2m30k) <= 0.5 and R(mag60) >= 0.8.
- P168 (a late-switched net keeps its routing): Rlab(c2m30k) <= 0.5, if testable.
- P169 (and its type): among seeds whose ctrl net is B type, at least half of their acquired c2m30k nets are B type, and at most a
  quarter of their acquired c2m2k nets.
- P170 (an A solution formed under mag survives the signed format): R(m2c10k) >= 0.5 and R(m2c30k) >= 0.5.
**Also reported (descriptive):** R and Rlab for every arm; the commitment step (the first switch step with R < 0.5); per-seed
tables; A/B counts, rom, xq and sign costs per arm; the trajectories, including how fast KL, lab and type change after a switch;
the reproduction of E30's 90,000-step ctrl and mag nets (same initialisation, stream and schedule); the branch-point identity of
the checkpoints at K. ctrl's checkpoints over 12 seeds can test whether E31's and E35's single-lineage dynamics generalise; that
analysis would be registered separately.
**What the outcomes would mean.**
- P166 and P167 hold (with P168 or P169): selection. The solution type is set in a critical period around the reliability onset;
  afterwards the B solution persists without the signs, and part of the nets' distance from Bayes is path dependence. The next
  question is what decides it at the onset: the initialisation or the order of the data (a branching test with new episode
  streams at the switch steps).
- P166 holds and P167 fails with R(c2m30k) high: no lock-in. The sign channel costs while it is present; the E30 gain is an input
  effect.
- P167's first part holds but R(mag60) < 0.8: the remaining budget may explain the late-switch deficit.
- P166 fails: 2,000 steps of signed training already shape the solution (commitment before the onset), or the switch itself costs.
- P170 holds: whichever solution forms first persists; an A solution is stable under the signs. P170 fails: the signed format keeps
  pushing toward a higher KL.
- Late-switched nets with B-like routing (P168) but a mag-like KL: the routing marker does not carry the cost.


## Amendment 46 (2026-10-30 02:05Z): E38, does size close the gap between the signed and the sign-free formats? (GPU)

Written before any E38 training. Known beforehand: as amendment 45, plus:
- E21: in 15 noise-free nets of widths 64-256 (pfn2, natural length), the big-miss departure persists and does not fall steadily
  with width; both middle-range patterns occur;
- E17: in the heads cohort, depth 3 reduced the departure to 0.40 of the 2-layer nets'; more or wider last-layer heads did not;
- E26.4 (descriptive; selected cohorts of 4 two-layer and 5 three-layer mixture nets): the 3-layer nets have about half the KL,
  and their signs-and-assignment cost is lower by less (0.0009 against 0.0012 nats), so its share of KL is not lower.
Plumbing as amendment 45 (the same scripts; width and depth are pfn2 arguments, and E21 trained these widths with the same code).
**Link to M2 (across sizes; the inductive bias).** E30's 2-3x gain from the sign-free format was measured at one size. If larger
nets learn the sign invariance from the signed format, the gap closes with capacity and the deviation is a small-model effect. If
it persists, it is an inductive bias of this architecture and recipe. The pooling account (E17, E20) says that a 2-layer net pools
a source's raw signed errors in block 1, before any per-token nonlinearity can make them even. It predicts that depth, which lets
a per-token transform precede a later pooling, closes the gap, and that width does not. E26.4's descriptive depth comparison
points the other way for the sign cost.
**Design** (`gpu/jobs_e38.txt` from `gpu/make_jobs.py`; `train_sw.py`, `e37_eval.py`, `e38_summ.py`).
- ctrl and mag (pure arms, amendment 45's fixed 90,000-step schedule), seeds 0-9, in three configs: width 64 and depth 2; width
  128 and depth 2; width 32 and depth 3 (2 heads and mixture head K = 4 throughout). Within a config and seed, both formats share
  the initialisation and the episodes (the stream depends on the config name). The width-32 depth-2 reference is E37's ctrl and
  mag arms (seeds 0-11).
- One on-demand L4 VM: 60 trainings, 8 at once, ordered by seed, deadlines as amendment 45; estimated $3.4. Checkpoints at 2,000,
  3,000, 5,000, 10,000, 20,000, 40,000, 60,000, 80,000 and 90,000 steps.
- Measures (`e37_eval.py`): K, rom and type, every block's query-row attention mass on the labelled tokens (the last block's is
  the reported marker), xq, and the sign-flip KL reduction for ctrl nets.
- Per config c: r(c) = mean K_mag / mean K_ctrl over seeds acquired in both (E30's paired ratio of means) and the reduction
  rho(c) = 1 - r(c); 2,000 bootstrap resamples of seeds.
**Predictions.**
- P171 (width does not close the gap): rho(w64) >= 0.3 and rho(w128) >= 0.3.
- P172 (depth closes at least half of it): rho(d3) <= 0.5 x rho(w32).
**Also reported (descriptive):** K_ctrl and K_mag per config (does the sign-free format gain from size?); A/B counts; the ctrl
nets' sign cost share per config; lab and xq per config; per-seed ratios.
**What the outcomes would mean.**
- P171 holds: the gap is not a small-width effect. With E21, the invariance failure is an inductive bias of the 2-layer recipe
  across widths. P171 fails: capacity learns the invariance.
- P172 holds: what this recipe lacks is a per-token stage before pooling, and the account predicts the size of the departure from
  the pooling order. P172 fails (rho(d3) close to rho(w32)): an extra per-token stage does not fix the invariance failure; the
  pooling-order account explains the big-miss shape (E17) but not the sign dependence.


## Amendment 47 (2026-10-30 02:05Z): E39, signs plus a magnitude feature, paired at initialisation (GPU)

Written before any E39 training. Known beforehand: as amendment 45, and E30's sgnmag arm (not paired at initialisation; natural
length): sign-flip reduction 1.3% of its KL against ctrl's 19.9%; KL 1.31x mag's on paired seeds [0.64, 2.54]; 3 of 10 acquired
nets B type (KL 0.009-0.022); 2 seeds stalled. Plumbing: `train_sw.py --pair_init` on CPU copies every shared parameter of the
ctrl initialisation of the same seed bit for bit; the extra input column lies in [-1/6, 1/6].
**Link to M2.** E30's disclosed confound was that sgnmag started from other weights than ctrl and mag. Paired at initialisation,
the comparison separates two readings of the signed format's cost:
- (a) computing an even function of a signed input with these layers is hard, so giving the magnitude directly removes the cost
  even with the signs visible (sgnmag about equal to mag);
- (b) the visible signs open a competing solution, so some seeds still select it (sgnmag between mag and ctrl).
With E37, this says whether selection at the onset depends on the even statistic being available.
**Design.** sgp: the sgnmag format (signed readings plus |o - y| on labelled tokens), seeds 0-11, the fixed 90,000-step schedule.
Every shared parameter comes from the ctrl initialisation of the same seed; the extra input column is drawn U(-1/6, 1/6) from a
torch generator seeded 31337 + seed; the episodes are E37's. Trained on E37's two VMs (six per VM), with E38's sparse checkpoints.
Compared with E37's ctrl and mag nets of the same seeds. Measures as amendment 45, including the sign-flip reduction.
r = mean K_sgp / mean K_mag over seeds acquired in both; c_sign as in E30.
**Prediction.**
- P173 (E30's P151, now paired): r <= 1.1 and c_sign(sgp) <= 0.25 x c_sign(ctrl).
**Also reported (descriptive):** A/B counts, KL against ctrl, lab and xq, and per seed against E30's unpaired sgnmag nets.
**What the outcomes would mean.** P173 holds: easy access to the magnitude is enough; the signed format's cost is the difficulty
of the even computation (reading a), and E30's failure came from the unpaired initialisation. P173 fails on KL with B-type sgp
nets: the visible signs keep a competing solution open even when the magnitude is given (reading b).


## Amendment 48 (2026-10-30 02:21Z): E40, do the sign-invariance failure and the gain from building the invariance in carry over to a second world? (GPU)

Written before any E40 training. Known beforehand: E1-E36 and E30 with their reviews; amendments 45-47 (E37-E39 registered and
training; no E37-E39 result seen). The new world and its plumbing:
- `t2_world.py`: in-context linear regression with an unknown noise level. w ~ N(0, I_D); v uniform over (0.1, 0.5, 2.0); N = 8
  context examples x_i ~ N(0, I_D), y_i = w.x_i + N(0, v); a query x*. Exact Bayes is a 3-component mixture (ridge per level,
  weighted by the marginal likelihood). It matches brute-force Gaussian conditioning (largest error 4e-15) and is exactly
  invariant under per-example flips (x_i, y_i) -> (-x_i, -y_i). On 20,000 episodes the level-blind reference is 0.178 (D = 1) and
  0.165 (D = 2) nats worse than Bayes, the scale of the main world's KL before acquisition (about 0.2).
- Smoke trainings on CPU (6,000 steps, seed 0; not part of the results): test gaps D = 1 ctrl 0.027, canon 0.022; D = 2 ctrl
  0.079, canon 0.064. On 6,000 fresh episodes the ctrl smoke nets' sign-flip KL reduction was 0.0049 of 0.025 nats (D = 1) and
  0.016 of 0.070 (D = 2), about 20-23% of their KL at that stage; the joint negation's was 0.0033 and 0.0062. At 6,000 steps
  (validation curves) the canonical format learned little faster than ctrl, unlike the main world's smoke runs (ctrl 0.078, mag
  0.021 at 5,000 steps).
- `e40_eval.py`: a first version used a grid centred on the Bayes mean and had minimum grid mass 0.71-0.91 on the smoke nets. It
  was replaced, before any registered net existed, by a per-episode grid spanning +-9 sd of every component of Bayes and of every
  density in the family (2,001 points, as pfn2.kl_grid); the smoke nets' minimum mass is then 1.0. `e40_summ.py` ran on
  synthetic inputs.
**Link to M2 (across tasks).** Every result so far comes from one world. The working account says that part of the nets' distance
from Bayes is a failure to learn posterior invariances that need an even function of each token's signed content, and that
building the invariance into the input gains more than the invariance cost measured on trained nets (E30: 3.2x). Task 2 differs
in structure: one regression problem instead of three sources, no labels or identity codes, exchangeable examples, and a
predictive that depends on the data through X^T X, X^T y and y.y. It shares the abstract property: the posterior is invariant to
flipping any example, and a linear read of a signed example is odd. If the failure and the amplified gain reappear here, they
are a property of how these transformers learn invariances, not of the first world. If they do not, the account must say what
distinguishes the worlds.
**Design** (`t2_world.py`, `train_t2.py`, `gpu/gpu_jobs_t2.sh`, `gpu/launch_t2.sh`, `gpu/retry_t2.sh`, `gpu/jobs_e40.txt`,
`e40_eval.py`, `e40_summ.py`; `pfn2.py` unchanged).
- pfn2's set transformer (2 layers, width 32, 2 heads, mixture head K = 4; no positional encoding) with input width D + 2;
  pfn2's AdamW, clipping and lr warmup; amendment 45's fixed 90,000-step schedule; seeds 0-11. Initialisation torch seed = seed,
  episode stream np.random.default_rng([seed, 2, 7]); both formats of a seed share both.
- Formats: ctrl (context tokens [x_i, y_i, 0], query [x*, 0, 1]); canon (each example replaced by its flip-orbit representative
  with y_i >= 0).
- Worlds: D = 2 (primary) and D = 1 (secondary), N = 8.
- One on-demand L4 VM: 48 trainings, 8 at once, ordered by seed; deadlines as amendment 45; estimated $2.5. Sparse checkpoints.
- Measures (`e40_eval.py`): test KL (gap on 16,000 episodes, seed 20,002), also by true noise level. On 6,000 fresh episodes (seed
  40,040), for ctrl nets: the KL reduction from geometric averaging over the base and 8 random per-example flip patterns (c_sign,
  uncorrected, as E27), and over the base and its joint negation (c_neg).
- Acquired: gap < 0.5 x the level-blind reference's gap. rho = 1 - mean K_canon / mean K_ctrl over seeds acquired in both (E30's
  paired ratio of means); 2,000 bootstrap resamples of seeds.
**Predictions** (D = 2; the account generalises).
- P174 (the nets depend on what Bayes ignores): c_sign >= 0.05.
- P175 (building the invariance in gains more than its measured cost): rho >= 1.5 x c_sign and rho >= 0.1.
**Also reported (descriptive):** the same quantities for D = 1; KL by noise level; c_neg; per-seed ratios.
**What the outcomes would mean.**
- P174 and P175 hold: the sign-invariance failure and its amplified retraining gain are not specific to the first world; the
  account gains a second task.
- P174 holds and P175 fails (rho near c_sign or below): the invariance cost appears, but building the invariance in gains only
  that cost here; E30's amplification is specific to the first world (for example to its solution-type selection).
- P174 fails: these nets learn the flip invariance in task 2 (attention's cross-token products may give the needed even statistics
  directly); the account must say why the first world differs.


## Amendment 49 (2026-10-30 02:24Z): E41, does E20's single-source pooling toy reproduce the sign-format amplification and lock-in? (CPU)

Written before any E41 training. Known beforehand: E20 (the 'ln' toy: one source's four labelled errors, linear token
embedding, LayerNorm, one learned-query attention pool, readout MLP to the three noise levels; 30,000 steps at constant lr; final
validation gaps 0.0058, 0.0015 and 0.0014 for seeds 0-2, so seeds can differ about fourfold); E30; amendments 45-48 (no E37-E40
result seen). Plumbing: `toy_fmt.py` on an untrained toy gives KL 0.62 and sign and negation costs below 0.0002 nats (the
untrained toy barely depends on the signs); 300 training steps take about 1.2 s; `e41_summ.py` ran on synthetic inputs.
**Link to M2 (why they cannot learn better; a minimal model).** If a model with only a source's own pooling step (no merge across
sources, no block-2 routing, no A/B types) shows the sign dependence, the amplified gain from a sign-free input and the lock-in,
then the mechanism lives in the pooling of signed per-token features, and the toy is small enough to analyse. If the toy shows
the dependence but not the amplification or the lock-in, those need the multi-source structure of the full nets.
**Design** (`toy_fmt.py`, `runq_e41.sh`, `e41_summ.py`; `minimal.py` unchanged). E20's 'ln' toy with minimal.py's
initialisation and episodes for each seed; formats ctrl (o = y + r) and mag (o = y + |r|); AdamW (lr 1e-3, weight decay 0.01),
batch 512, clipping 1; constant lr then a linear cooldown. Arms, seeds 0-11:
- ctrl, mag: 24,000 + 6,000 steps;
- c2m2k, c2m12k: ctrl, then mag after 2,000 or 12,000 steps (same schedule);
- mag18: mag with 12,000 + 6,000 steps, c2m12k's budget after its switch;
- m2c12k: mag, then ctrl after 12,000 steps.
Evaluation on 20,000 fresh episodes (seed 4141) in the final format, against the exact posterior over the three levels: K = mean
KL; for ctrl-format toys, the KL reduction of the normalised geometric mean of the class probabilities over the base and 8 random
sign patterns (c_sign = pooled reduction / pooled family KL, uncorrected) and over the base and its negation (c_neg). Also KL at
steps 1,000-30,000 (descriptive). rho = 1 - mean K_mag / mean K_ctrl; R(a) as amendment 45; 2,000 bootstrap resamples of seeds.
At most 4 processes, nice 19, ionice -c3.
**Predictions.**
- P176 (the toy depends on the signs): c_sign >= 0.05.
- P177 (the sign-free input gains more than the measured sign cost): rho >= 1.5 x c_sign.
- P178 (lock-in in the toy): R(c2m2k) >= 0.8, R(c2m12k) <= 0.5 and R(mag18) >= 0.8.
**Also reported (descriptive):** c_neg; R(m2c12k); per-seed KLs; the KL trajectories.
**What the outcomes would mean.**
- P176-P178 hold: single-source pooling of signed features reproduces the sign dependence, the amplified gain and the lock-in, so
  the full nets' versions can be studied in a model small enough to analyse.
- P176 holds, P177 fails: the toy pays only the sign cost; the amplification in the full nets needs their multi-source
  structure (for example the A/B solution types).
- P177 holds, P178 fails: the sign format makes the toy's solution worse without lock-in; a late switch recovers.


## Amendment 50 (2026-10-30 03:09Z): E42, do E31's and E35's single-lineage training dynamics generalise across lineages? (CPU, on E37's ctrl checkpoints)

Written before E37's checkpoints are copied back and before any E42 quantity of a registered lineage is computed. Known beforehand:
- E31 (one lineage per type): in the A replay (base_mix4_s6) the sign-flip cost and block 1's cross-source same-question attention
  xq fall together at 12,000-20,000 steps (window ratios 0.37 and 0.15); in the B replay (base_mix4_s5) they stay (0.74 and 0.96).
- E35: block 2's query attention to the labelled tokens rises in the B replay after acquisition (0.09 at 3,000 steps, 0.24 at
  10,000, 0.38 at the end) and falls modestly in the A replay (0.36 at 7,500 to 0.26); P162 (a late fall in A) failed.
- E37's ctrl arm (amendment 45) retrains seeds 0-11 under the fixed schedule with the original initialisation and stream. Seeds
  5 and 6 are E16's replays (same initialisation and stream), so they are excluded. For seeds 0-4 and 7 the final states were
  measured on the original bases (E34 block-2 labelled mass: s0 0.069, s1 0.280, s2 0.071 (A); s4 0.424, s7 0.401 (B)), so only
  their trajectories are new; seeds 8-11 are new lineages. E30's ctrl types: A for seeds 0, 1, 2, 6; B for 4, 5, 7-11; seed 3
  stalled.
- E41 (Update 47) and the partial view of E37's validation logs while checking that the VM ran (seed 6 at about 60,000 steps;
  constant-lr validation gaps only, no E42 quantity).
Plumbing (disclosed): `e42_run.py` on E16's replay checkpoints (s5 and s6 at 10,000 and 20,000 steps, through check-only path
overrides) reproduces E31's and E35's values: s5 10,000: flip cost 0.0108, KL 0.053, xq 0.058, labelled mass 0.237; s6 20,000:
0.0018, 0.031, 0.005, 0.274.
**Link to M2 (how they learn).** E31 and E35 described how one A and one B lineage form their solutions: the A lineage loses its
sign dependence and its block-1 cross-source attention late (12,000-20,000 steps); the B lineage keeps both, and builds up block
2's direct read of the labelled tokens after acquisition. If this holds across lineages, the A/B difference is a difference in a
late refinement stage, and the B solution's persistence is general; if not, the single-lineage story was an accident.
**Design** (`e42_run.py`, `e42_summ.py`).
- Lineages: E37's ctrl arm, seeds acquired (final gap < 0.10) except 5 and 6; type from the final net's middle-range rom (E37's
  evaluation).
- Checkpoints 3,000, 10,000, 12,000, 15,000, 20,000, 25,000, 30,000, 40,000, 60,000 and 90,000 steps. Measures exactly as E31 on
  E28's first 3,000 bases (sign-flip and negation costs, 9 samples, uncorrected; xq), plus E35's block-2 labelled and summary
  masses.
- Per lineage, r_flip and r_xq = median over 20,000-30,000 steps divided by the median over 10,000-15,000 (E31's windows).
**Predictions.**
- P179 (E31's P154 generalises): the median over A-type lineages of r_flip is <= 0.5, and the median over B-type lineages >= 0.7.
- P180 (E31's P155 generalises): the same for r_xq.
- P181 (E35's B pattern generalises): in at least two thirds of B-type lineages, the block-2 labelled mass at 90,000 steps exceeds
  that at 3,000.
- P182 (E34's end-state separation in new lineages): every B-type lineage among seeds 8-11 ends with labelled mass >= 0.30 (above
  every A base's in E34).
**Also reported (descriptive):** per-lineage trajectories; the negation/flip cost ratio over training; the timing of the xq fall.
**What the outcomes would mean.**
- P179 and P180 hold: the A type's late refinement (losing the sign dependence and the cross-source attention together) is a
  general stage of this recipe, not one lineage's accident. They fail: E31's pattern was lineage-specific.
- P181 and P182 hold: B's block-2 read of the labelled tokens is generally built after acquisition and separates the types.


## Amendment 51 (2026-10-30 03:14Z): E43, what fixes a seed's solution type, and when? Data-order branching and initialisation swaps (GPU)

Written before any E43 training. Known beforehand: E1-E36, E30, E41 (Update 47), amendments 45-50 (E37-E40 and E42 registered; no
E37-E40 result seen beyond VM health checks, which showed seed 6's constant-lr validation gaps). In particular:
- E30's GPU-trained ctrl nets reproduce the original CPU-trained bases (same initialisation and stream) to 3-4 digits: this
  training is numerically stable, and a seed's type is reproducible given its initialisation and episode stream;
- E16 (one lineage per type): the A lineage starts with B's middle-range sign and turns A-like late, with back-and-forth; E31: its
  sign dependence and cross-source attention fall at 12,000-20,000 steps; E35: B's block-2 labelled attention rises after
  acquisition;
- E30's ctrl types: A for seeds 0, 1, 2, 6; B for 4, 5 and 7-11; seed 3 stalled.
Plumbing (disclosed): `train_br.py` on CPU (seed 5): a run branched at 500 steps equals E16's replay dyn_base_s5 bit for bit at
steps 0 and 500 and differs at 1,000; an initialisation swap starts from other weights. `e43_eval.py` ran on that smoke net and
`e43_summ.py` on synthetic inputs.
**Link to M2 (how they learn; predicting the departure).** A net's A/B type decides much of its distance from Bayes (E26, E30).
If the random initialisation fixes the type, the type can be predicted before training, and finding the initial property that
predicts it becomes the next step toward predicting the departure from the architecture and its initialisation. If the early data
order fixes it, or both fix it jointly at the reliability onset, training chooses it, and E37's format switches say what the
choice depends on. The tool is instability analysis (Frankle et al. 2020, "Linear mode connectivity and the lottery ticket
hypothesis"): copies of a run that continue on new data from step K end in the same solution only once training has committed.
**Design** (`train_br.py`, `gpu/gpu_jobs_x.sh`, `gpu/launch_x.sh`, `gpu/retry_x.sh`, `gpu/jobs_e43a.txt`, `gpu/jobs_e43b.txt`,
`e43_eval.py`, `e43_summ.py`; `train_sw.py`'s model building and amendment 45's fixed 90,000-step schedule).
- Parents: E37's ctrl nets, seeds 0, 1, 2 and 4-11 (seed 3 stalled in E30). For each seed, 7 arms (ctrl format):
  - br0: the parent's initialisation and a new episode stream (np.random.default_rng([seed + 1000, cid, 7])) from step 1;
  - br2k, br3k, br5k, br10k, br20k: the parent's run up to K steps (same initialisation and stream, so bit-identical up to K),
    then the new stream;
  - newinit: a new initialisation (torch seed = seed + 1000) with the parent's stream.
- Two on-demand L4 VMs (42 and 35 trainings, 8 at once; deadlines as amendment 45); estimated $4. Sparse checkpoints, plus K.
- Measures (`e43_eval.py`, also on the parents): test KL; E15's rom and the type (middle-range sign); block 2's labelled mass and
  xq; the predictive mean on 4,000 test episodes.
- Quantities: agreement A(arm) = the share of acquired branch nets with their parent's type; similarity rho = the correlation of the
  branch's and the parent's deviations from the Bayes mean (baseline: different parents of the same type); chance agreement =
  p_B^2 + p_A^2 from the parents' type frequencies. 2,000 bootstrap resamples of seeds.
**Predictions** (selection at the reliability onset, the coordinator's hypothesis).
- P183 (the type is fixed by 5,000 steps): A(br5k) >= 0.9 and A(br10k) >= 0.9.
- P184 (neither the initialisation alone nor the data order alone fixes it): A(br0) <= 0.8 and A(newinit) <= 0.8.
**Also reported (descriptive):** A for br2k, br3k and br20k; flip directions; similarity and KL ratios by branch step; the
commitment step (the first K with A >= 0.9).
**What the outcomes would mean.**
- P183 and P184 hold: training chooses the type around the reliability onset, from the joint effect of initialisation and data.
- P184 fails because A(br0) is high: the initialisation fixes the type; the next step is to predict it from the initial weights.
- P184 fails because A(newinit) is high: the data order fixes it.
- P183 fails: the type is still open at 5,000-10,000 steps and is decided in the later refinement, as E16's A lineage suggested;
  br20k shows how late.


## Amendment 52 (2026-10-30 06:50Z): E44, which part of the initialisation carries the solution type? Hybrid initialisations (GPU)

Written before any E44 training. Known beforehand: E37-E43 with their reviews (Updates 47-53). In particular:
- E43: with the parent's initialisation and a new episode stream, 10 of 11 copies kept the parent's A/B type (median KL difference
  9%, deviation similarity 0.76 against 0.41 between unrelated same-type parents); with a new initialisation and the parent's
  stream, 5 of 8 acquired copies agreed (chance 0.54) and 3 of 11 stalled;
- E37: every net shows the B type at the first checkpoint (3,000 steps); without signs most leave it; E42: A-type attention
  signatures (low block-1 cross-source attention; in 2 of 3, low block-2 attention to the labelled tokens) appear by 10,000 steps;
- E34: block 2's query attention to the labelled tokens separates the types (A 0.06-0.28, B 0.39-0.45).
Parent types (E37's ctrl nets): A for seeds 0, 1, 2, 6; B for 4, 5, 7, 9, 10, 11 (and 8, excluded as an outlier whose switched nets
all stay bad, E37). Parameter groups that differ between seeds at initialisation: the input embedding, each block's attention
(qkv and output projection) and MLP, and the output head; the LayerNorm parameters are identical.
Plumbing (disclosed): `train_hy.py` on CPU (recipient seed 4, donor 0, group b2att, 500 steps): its step-0 weights equal seed 0's
initialisation in exactly the four block-2 attention tensors and seed 4's in the other 26. `e44_summ.py` ran on synthetic inputs
and recovered a planted group.
**Link to M2 (predicting the departure from the architecture and its initialisation).** E43 found that a net's type, and much of
its distance from Bayes, stays with its initial weights. If one group of initial weights carries the type, the departure can be
traced to a specific architectural component's starting state: for example block 2's attention, whose learned routing separates
the types (E34), or the input embedding, which sets how errors and identity codes enter. If no single group carries it, the type
is a joint property of the initialisation.
**Design** (`train_hy.py`, `gpu/gpu_jobs_x.sh`, `gpu/launch_x.sh`, `gpu/retry_x.sh`, `gpu/jobs_e44a.txt`, `gpu/jobs_e44b.txt`,
`e43_eval.py`, `e44_summ.py`; `train_sw.py`'s model building and amendment 45's fixed 90,000-step schedule, ctrl format).
- Six A/B parent pairs: (0, 4), (1, 5), (2, 7), (6, 9), (0, 10), (1, 11).
- For each pair and each of six groups (inp; b1att; b1mlp; b2att; b2mlp; head), two hybrids: the B parent's initialisation with the
  group from the A parent, and the A parent's initialisation with the group from the B parent; each trained on its recipient's
  episode stream. 72 trainings on two on-demand L4 VMs (36 each, 8 at once; deadlines as amendment 45); estimated $3.5. Sparse
  checkpoints.
- Measures (`e43_eval.py`): test KL, rom and type (middle-range sign), block-2 labelled attention, xq, and the predictive mean on
  4,000 test episodes; similarity of each hybrid's deviation from the Bayes mean to the donor parent's and to the recipient
  parent's.
- T(G) = the share of acquired hybrids of group G whose type equals the donor parent's (both directions pooled; 12 per group);
  2,000 bootstrap resamples of the six pairs.
**Predictions.**
- P185 (one group carries most of the type): the largest T(G) over the six groups is >= 0.75.
- P186 (that group is block 2's attention): the group with the largest T is b2att.
**Also reported (descriptive):** T and the A-to-B and B-to-A transfer counts per group; similarity to the donor and to the
recipient; KL and stalls per group.
**What the outcomes would mean.**
- P185 and P186 hold: the type is carried by block 2's initial attention weights, the component whose learned routing separates
  the types; the next test predicts the type from properties of those weights.
- P185 holds with another group: that component's initial state carries the type.
- P185 fails with all T low: no single group carries it; each hybrid mostly follows its recipient (T near 0) or the outcome is
  mixed (T near 0.5); the type is then a joint property of the initialisation, and partial swaps of two or more groups are next.


## Amendment 53 (2026-10-30 06:54Z): E45, are single-pass LLM answers invariant to negating a source's past errors? (LLM, about $5)

Written before the E45 items file exists and before any experimental call. Known beforehand: E18 (reasoning LLMs given the model
track Bayes and compute with sums of squares; in the unstated "forecasters" framing Sonnet's answers moved with a forecaster's
average error, which it tried to correct for) and E18b (single-pass answers, thinking off, are much noisier and show no consistent
big-miss departure; per-call list costs: Haiku 4.5 about $0.0005, Opus 4.8 $0.0025, Sonnet 4.6 $0.005; GPT and local Qwen free).
E27-E29 and E40: the small nets depend on the signs of the labelled errors (E28: not as a proportional mean-error correction).
Plumbing (disclosed): `e45_data.py` ran to a scratch path (60 pairs; Bayes identical within every pair; the negated source's
|mean error| median 0.65, max 1.91); one live call each to Haiku 4.5 and GPT-5.6 on a dummy averaging prompt (not an item) answered
"0.52" with 0 thinking tokens (Haiku $0.00015); `e45_summ.py` ran on synthetic answers and recovered planted slopes.
**Link to M2 (pretrained LLMs).** M2 asks for an account that also covers pretrained LLMs. The small nets fail to learn the
posterior's sign invariance (E27-E30, E40). Do single-pass LLMs, which know the task only from text, treat a source's error signs
as Bayes does when told that the errors have mean zero? A natural non-Bayesian heuristic is bias correction: subtract a source's
average past error from its reading. That would be a sign dependence driven by a learned prior about biased forecasters, unlike
the nets' (E28 found theirs is not a proportional mean-error correction). Reasoning removed most departures in E18; this asks
what a single forward pass does.
**Design** (`e45_data.py`, `e45_call.py`, `e45_cap.sh`, `e45_summ.py`; E18b's call functions in `e18b_call.py` unchanged).
- Items (`results/e45_items.json`, generated after this hash and then hashed): from each of E18's 60 base pairs, the three sources
  of the 'spread' version as displayed in E18; the negated source is the one with the largest |mean error|. Versions: orig, and
  neg with that source's four displayed errors multiplied by -1. Bayes is the same in both. Prompts: E18's stated and unstated
  templates and source labels.
- Arms: S_ and U_ (stated, unstated) x haiku45, sonnet46, opus48 (claude -p, thinking off, 16-token cap, E18b's system instruction),
  gpt56 (codex-any gpt-5.6-sol, reasoning none, tools disabled), qwen17 (local Qwen3-1.7B, greedy). Two samples per prompt, one for
  qwen17. About 1,440 Claude calls; API spend cap $6 at list price (`e45_cap.sh` stops the API streams if passed).
- Measures (`e45_summ.py`): per arm and pair, the shift D = mean answer(neg) - mean answer(orig) and X = 2 x mean error x Bayes
  weight of the negated source in orig; the bias-correction slope c (OLS of D on X without intercept; c = 1 for a full correction);
  the invariance ratio mean D^2 / retest variance (about 1 under invariance); the mean absolute deviation from Bayes. Pooled = mean
  over the three Claude arms of a framing. 2,000 bootstrap resamples of pairs.
**Predictions.**
- P187 (even with the zero-mean model stated, single-pass LLMs treat a source's mean error as a bias): stated framing, the pooled
  slope c over the Claude arms has a 95% interval above 0.
- P188 (without the model, they correct for most of it): unstated framing, the pooled slope c over the Claude arms is >= 0.5.
**Also reported (descriptive):** per-arm slopes, invariance ratios and deviations, including gpt56 and qwen17.
**What the outcomes would mean.**
- P187 holds: single-pass LLMs are not sign-invariant here; their dependence takes the form of a bias correction learned from
  text, a different mechanism from the small nets' (an account covering both must allow semantic priors as well as architectural
  failures).
- P187 fails with c near 0 and invariance ratios near 1: their single-pass answers are sign-invariant within noise, unlike the nets.
- P188 holds and P187 fails: stating the model is enough to switch the heuristic off.


## Amendment 54 (2026-10-30 06:58Z): E46, is the map from initial weights to solution type smooth? Perturbed initialisations (GPU)

Written before any E46 training. Known beforehand: E43 (the A/B type stays with the initialisation under a new episode stream in
10 of 11; a new initialisation gives about chance), E37 (GPU training here is bit-deterministic, and the GPU nets reproduce the
CPU-trained bases), amendments 52-53 (E44 and E45 registered and running; no E44 or E45 result seen).
Plumbing (disclosed): `train_pt.py` on CPU (seed 4, eps 0.01, 500 steps): every randomly initialised tensor of the step-0
weights differs from seed 4's initialisation by a relative standard deviation of 0.0098-0.0107, and the LayerNorm parameters are
unchanged. `e46_summ.py` ran on synthetic inputs.
**Link to M2 (predicting the departure from the initialisation).** If the type changes only under large perturbations, nearby
initialisations share it, and predicting it from properties of the initial weights is plausible (E44 asks which weights). If
perturbations of 0.3% already flip it, the map from initialisation to type is effectively chaotic at that scale, though training
is deterministic: the type is then predictable only by running training, which bounds what "predicting the departure from the
architecture and its initialisation" can mean.
**Design** (`train_pt.py`, `gpu/gpu_jobs_x.sh`, `gpu/launch_x.sh`, `gpu/retry_x.sh`, `gpu/jobs_e46a.txt`, `gpu/jobs_e46b.txt`,
`e43_eval.py`, `e46_summ.py`; amendment 45's fixed 90,000-step schedule, ctrl format).
- Parents: E37's acquired ctrl nets (seeds 0, 1, 2, 4-11).
- For each parent, initialisations theta + eps x std(theta) x xi for every randomly initialised tensor (xi ~ N(0, I) from a torch
  generator seeded 50,000 + 100 x seed + draw; LayerNorm parameters unchanged), eps in {0.003, 0.03, 0.3}, two draws each; the
  parent's episode stream. 66 trainings on two on-demand L4 VMs (33 each); estimated $3. Sparse checkpoints.
- Measures (`e43_eval.py`): test KL, type (middle-range rom sign), block-2 labelled attention, xq, predictive mean; agreement A(eps)
  with the parent's type; similarity of the deviation from the Bayes mean to the parent's; median |KL / parent KL - 1|. 2,000
  bootstrap resamples of parents.
**Predictions.**
- P189 (the type is locally stable in the initial weights): A(0.003) >= 0.9.
- P190 (a large perturbation acts like a new initialisation): A(0.3) <= 0.8.
**Also reported (descriptive):** A(0.03); similarity and KL differences by eps, against E43's br0 (0.76; same initialisation, new
stream) and newinit (0.34).
**What the outcomes would mean.**
- P189 and P190 hold: the type is a locally smooth function of the initial weights with a finite basin; looking for initial
  properties that predict it (E44 and after) is well posed.
- P189 fails: a 0.3% perturbation already changes the type in more than one parent in ten: the map is effectively chaotic at that
  scale, and the type cannot be read off coarse properties of the initialisation.
- P190 fails: even a 30% perturbation keeps the type; the type depends on coarse properties that survive large noise.


## Amendment 55 (2026-10-30 07:50Z): E47, when does a net's final A/B type become visible in training? (CPU)

Written before any E47 quantity is computed on any checkpoint. Known beforehand: E37-E45 with their reviews; the final evaluations
of E43's branch nets (types and KL; Update 50); not yet E44 or E46. In particular:
- E37's ctrl trajectories (descriptive): at 3,000 steps every net has a positive middle-range rom (B); block-2 labelled attention
  medians 0.29-0.37 over training; E42: at 3,000 steps the labelled attention is 0.032, 0.355, 0.023 in the three new A lineages
  and 0.25-0.38 in the B lineages; xq at 3,000 steps 0.045-0.089 (A) and 0.04-0.09 (B);
- E35 (two replays): at 1,000 steps the B replay's labelled attention (0.165) was below the A replay's (0.348);
- E43: the type stays with the initialisation under a new episode stream (10 of 11).
Plumbing (disclosed): `e47_run.py` on two smoke nets (through a check-only output override): at step 0 the block-2 labelled mass is
0.750 and xq 0.137 (uniform attention gives 0.75 and 0.133). `e47_summ.py` ran on synthetic inputs and recovered a planted marker.
**Link to M2 (how they learn; predicting the departure).** E43 says the initial weights carry the type. If a simple functional
marker separates the eventual A and B nets early in training, the choice is expressed early and can be read before the solution
is complete; together with E44 (which weights) and E46 (how smooth), this traces the type from initialisation to function.
**Design** (`e47_run.py`, `e47_summ.py`).
- Lineages: every run trained in ctrl throughout under amendment 45's fixed schedule at width 32 and depth 2 and acquired at the
  end: E37's ctrl arm (12 seeds), E43's branch and initialisation-swap nets, E44's hybrids and E46's perturbed nets (the last two
  when they finish). The final type is each run's middle-range rom sign. An E43 brK run is left out at steps <= K, where it
  duplicates its parent.
- Markers at steps 0, 2,000, 3,000, 5,000 and 10,000: block-2 query attention to the labelled tokens (lab), block-1 xq, the
  sign-flip cost share (E28's 9-sample family on 1,000 bases), and test KL (4,000 episodes).
- AUC = P(marker of a final-B lineage > marker of a final-A lineage); 2,000 bootstrap resamples of clusters (the parent seed:
  E37 seed, E43 parent, E44 recipient, E46 parent).
**Predictions.**
- P191 (the B route is visible early): AUC(lab at 3,000 steps) >= 0.75.
- P192 (the type is not visible in the initial function): at step 0, |AUC - 0.5| <= 0.15 for every marker.
**Also reported (descriptive):** every marker's AUC at every step; the earliest step at which some marker reaches 0.8.
**What the outcomes would mean.**
- P191 holds: the B route (block 2 reading the labelled tokens) separates the eventual types by 3,000 steps, early in the
  formation of reliability weighting.
- P191 fails: the eventual type is not visible in that route at 3,000 steps, although E43 says it is already set; it is carried by
  something these markers miss.
- P192 fails: some function of the initial net already separates the types, a lead for predicting them from the initialisation.


## Amendment 56 (2026-10-30 09:37Z): E48, is the format gain about linear sufficient statistics rather than the invariance? A 2x2 format test in both worlds (GPU)

Written by the fifth agent before any E48 training and before any E48 quantity is computed on a registered net. Known
beforehand: E1-E47 with their reviews (Updates 10-57). In particular:
- World 1 (fixed 90,000-step schedule, seeds 0-11; E37, E39): ctrl mean KL 0.0127 (11 acquired; 4 A, 7 B; large-range rom
  0.133; sign-flip cost share 0.200); mag (labelled readings y + |r|) 0.0055 (12 acquired; 11 A; rom 0.073); sgp (signed
  readings plus a feature |r|, paired at initialisation) 0.0082 (9 acquired, 3 stalls, 2 B; sign share 0.008).
- World 2 (E40): D = 2 ctrl 0.0187, canon 0.0092 (reduction 0.51; ctrl sign share 0.34); D = 1 ctrl 0.0032, canon 0.0033 (no
  gain; sign share 0.32). Post review: at D = 1 the ctrl nets' sign-averaged predictions (0.0019) beat the canonical nets (0.0029)
  on identical episodes.
- E41: E20's single-source pooling toy gains nothing from mag. E19: a per-token MLP before the first attention lowered the KL of
  the acquired B seed from 0.0101 to 0.0045 and of the A seed from 0.0070 to 0.0061 (one seed stalled). E20: in the toy,
  LayerNorm followed by mean pooling undercounts a single big miss; an oracle log-SS readout has no departure.
Plumbing (disclosed; not results):
- `train_lin.py` (train_sw.py with the format module `e48_fmt.py`) and `train_t2lin.py` (train_t2.py's recipe with `t2lin.py`'s
  formats) reproduce train_sw.py and train_t2.py bit for bit on CPU after 500 ctrl steps (seed 0; world 2 at D = 2);
- sq's step-0 weights equal E37 ctrl s0's; sgnsq's (--pair_init) equal E39 sgp s0's; prod's and sgnprod's are identical, and
  every parameter they share with ctrl equals E40 d2ctrl s0's step 0, with the new input columns within +-1/sqrt(D + 2);
- the token builders match direct formulas (the products summed over examples equal X^T X, X^T y and y.y; prod tokens are
  invariant under per-example flips; sq's labelled reading minus label equals r^2) (`e48_plumb.py`, `e48_plumb.sh`);
- CPU smoke runs, seed 0, 3,000 constant-lr steps (validation gaps; not results): sq 0.095, sgnsq 0.064; D = 2 prod 0.075,
  sgnprod 0.076; D = 1 prod 0.037 (best 0.035 at 2,000), sgnprod 0.043 (`e48_smoke.sh`); `e48_eval_w1.py` and
  `e48_eval_t2.py` ran on these nets; `e48_summ.py` ran on synthetic inputs (`e48_synth.py`) and recovered the planted ratios,
  a planted stall and the planted decisions.
**Link to M2 (across tasks; predicting the departure from the architecture).** Building the sign invariance into the input
gained 3.2 times its measured cost in world 1 (mag), about 1.5 times at D = 2 and nothing at D = 1 in world 2 (canon), and less
than nothing in the toy (E30, E40, E41). An account derived from the architecture should explain this. In these pre-norm nets,
block 1's attention pools linear functions of each token's LayerNormed features, and LayerNorm is the only per-token
nonlinearity before that pooling. The posterior's sufficient statistics are sums of per-token nonlinear statistics (world 1: r^2
for each labelled token, summed per source; world 2: x x^T, x y and y^2 for each example). The account under test (A-lin): the
departure, and the gain from a format, are set mainly by how linearly the input delivers those per-token statistics, not by
whether a nuisance is removed:
- mag's |r| is a monotone function of the needed r^2, so pooled |r| is a usable proxy (a large gain); canon's folding leaves
  world 2's products nonlinear (a small gain or none);
- formats that give the per-token statistics linearly should gain more than the folds, in both worlds;
- once the statistics are linear, visible signs should cost little (as E39 found for |r|).
Alternatives: (A-inv) the sign channel costs in itself (an invariance failure, interference, or E37's solution-type selection),
so signs plus linear statistics stay costly; (A-cap) LayerNorm caps per-token evidence (E20's toy), so heavy-tailed linear
statistics (r^2, y^2, x y) help little and the big-miss departure stays or grows with r^2.
**Design** (`e48_fmt.py`, `train_lin.py`, `t2lin.py`, `train_t2lin.py`, `gpu/jobs_e48a.txt`, `gpu/jobs_e48b.txt`,
`gpu/jobs_e48c.txt`, `e48_eval_w1.py`, `e48_eval_t2.py`, `e48_summ.py`; `train_sw.py`, `train_t2.py`, `t2_world.py`, `e30_fmt.py`,
`e30_eval.py`, `e37_eval.py`, `e40_eval.py`, `pfn2.py` and `world.py` unchanged; GPU scripts `gpu/gpu_jobs_x.sh`,
`gpu/launch_x.sh`, `gpu/retry_x.sh`).
- Arms, amendment 45's fixed 90,000-step schedule, seeds 0-11, the same episodes as the existing arms of each seed:
  - world 1: sq (labelled readings y + r^2; the initialisation of E37's ctrl and mag nets, so sq differs from mag only in
    |r| -> r^2); sgnsq (ctrl's signed readings plus a feature r^2 on labelled tokens; --pair_init, so the initialisation of
    E39's sgp nets, from which sgnsq differs only in |r| -> r^2). The 2x2: ctrl (signs; not linear), mag (no signs; folded),
    sgnsq (signs; linear), sq (no signs; linear); sgp (signs; folded) for reference;
  - world 2, D = 2 and D = 1, N = 8: prod (context tokens carry only the per-example products x_a x_b for a <= b, x_a y and
    y^2); sgnprod (ctrl's [x, y] plus the same products); the query token [x*, 0, 1] in both. prod and sgnprod start from
    identical weights, equal to ctrl's in every shared parameter. The 2x2: ctrl, canon (E40), sgnprod, prod.
- GPU: three on-demand L4 VMs (world 1; world 2 at D = 2; world 2 at D = 1), 24 trainings each, 8 at once, sparse checkpoints;
  deadlines as amendment 45; estimated $3.4 at the $0.87/h ceiling. Every VM is logged in `GPU-LOG.md` and checked in the instance
  list after deletion.
- Measures (CPU, nice 19). World 1 (`e48_eval_w1.py`, e37_eval.py's measures): test KL in the net's format; rom (large and
  middle range) and the type; block-2 labelled attention and xq; the sign-flip cost share for sgnsq. World 2 (`e48_eval_t2.py`,
  e40_eval.py's measures): test KL, also by noise level; the sign-flip and negation costs for sgnprod.
- Acquired: world 1 gap < 0.10; world 2 gap < 0.5 x the level-blind reference's. r(a/b) = mean K_a / mean K_b over seeds
  acquired in both (E30's paired ratio of means); c_sign = pooled sign-flip KL reduction / pooled family KL over acquired nets
  (uncorrected). Intervals: 2,000 bootstrap resamples of seeds, with multiplicities.
**Predictions** (A-lin).
- P193 (world 2: linear statistics gain beyond the invariance): r(prod/canon) <= 0.6 at D = 2 and at D = 1.
- P194 (world 2: given linear statistics, visible signs cost little): at D = 2, r(sgnprod/prod) <= 1.25 and c_sign(sgnprod)
  <= 0.25 x c_sign(ctrl).
- P195 (world 1: the linear statistic gains beyond the fold): r(sq/mag) <= 0.8.
- P196 (world 1: the big-miss departure follows the shape of the per-token statistic): the mean large-range rom of sq is <= 0.5 x
  that of mag, over the seeds acquired in both.
- P197 (world 1: given the linear statistic, visible signs cost little): r(sgnsq/sq) <= 1.25 and c_sign(sgnsq) <= 0.25 x
  c_sign(ctrl).
**Also reported (descriptive):** each world's 2x2 as ratios to ctrl; stalls; A/B counts, lab and xq (world 1); sgnsq against
E39's sgp (identical initialisation; r^2 instead of |r|); KL by noise level and the negation cost (world 2); P194's parts at
D = 1; per-seed tables.
**What the outcomes would mean.**
- P193 and P195 hold: in both worlds, giving the posterior's per-token statistics linearly gains well beyond removing the signs.
  The departure is then mostly the cost of computing nonlinear per-token statistics before pooling, and a format's gain is set
  by how far it linearizes them. That explains the world difference: mag folds r into a monotone proxy of the needed statistic,
  while canon leaves world 2's products nonlinear. With P194 and P197, the sign dependence is a by-product of that computation
  rather than a separate failure to learn the invariance.
- P194 or P197 fails (signs still cost, or still give B types and stalls): the visible sign channel costs in itself (A-inv), on
  top of the difficulty of the statistic.
- P196 holds: the big-miss departure comes from the shape of the per-token feature (|r|, or a capped r, instead of r^2). It
  fails: LayerNorm capping or the pooling reintroduces it even when r^2 is given (A-cap).
- P193 fails at D = 1 only: at D = 1 the per-token products are not the bottleneck; the gain from a format then depends on where
  each task's bottleneck lies.
- P193 and P195 fail: the linear-pooling account does not explain the format gains, and world 1's large gain needs another
  explanation (for example E37's solution-type selection).


## Amendment 57 (2026-10-30 09:48Z): E49, does a per-token MLP before the first attention remove much of the departure in both worlds? (architecture; GPU)

Written by the fifth agent before any E49 training. E48 (amendment 56) is registered and training; no E48 result has been seen.
Known beforehand: E1-E47 with their reviews. In particular:
- E19 (natural-length recipe, seeds 5-7 of world 1): the same extra block before the first attention ("pre") lowered the KL of
  the acquired B seed from 0.0101 to 0.0045 and of the A seed from 0.0070 to 0.0061 and the big-miss departure partly (0.57 x
  pooled); seed 7 did not acquire within its natural-length run; the same block after the last block ("post", the parameter
  control) changed neither the departure nor the KL (0.0109, 0.0067, 0.0131 against 0.0101, 0.0070, 0.0141).
- E17, E38: depth 3 reduced the big-miss departure (E17) but left most of the sign-free format gap (E38: rho 0.57; ctrl sign
  share 15%); widths 64-128 did not close the gap (E38).
- E37 (world 1) and E40 (world 2) ctrl nets, the baselines here: world 1 mean KL 0.0127 (11 acquired), sign share 0.200; world
  2 D = 2 0.0187, sign share 0.341; D = 1 0.0032, 0.324.
Plumbing (disclosed; not results):
- `pfn_pre.py` builds pfn2's PFN plus the extra block, created after every standard module: in CPU smoke runs (seed 0) every
  shared parameter at step 0 equals E37 ctrl s0's (world 1) and E40 d2ctrl s0's (world 2) step 0, and the variant has exactly
  the extra block's six tensors in addition; with that block's output layer zeroed it computes exactly the baseline's function
  (`e49_plumb.py`);
- `train_pre.py` (train_sw.py and train_t2.py with only the model builder replaced) ran 3,000 constant-lr steps in each world
  (validation gaps at 3,000 steps: world 1 0.183, D = 2 0.072, D = 1 0.047; not results; `e49_plumb.sh`); `e49_eval.py` ran on
  these nets; `e49_summ.py` ran on synthetic inputs and recovered the planted ratios, a planted stall and the planted decisions.
**Link to M2 (predicting the departure from the architecture).** Amendment 56's account A-lin, stated for the architecture: in
these pre-norm 2-layer set transformers, block 1's attention pools linear functions of LayerNormed tokens, so the per-token
nonlinear statistics that the posterior needs (r^2; x x^T, x y, y^2) can only enter through LayerNorm, softmax weights and
bilinear scores, and the departure is the cost of that. E48 changes the input; E49 changes the architecture instead and keeps the
signed ctrl format: one token-wise MLP before the first attention can compute those statistics for each token before any
pooling. If the departure and the sign dependence then fall in both worlds, the departure is predicted by where the architecture
places its first per-token nonlinearity relative to its first pooling. If they do not fall (and E48's linear formats do help),
the architecture can represent the statistics but training does not find the per-token computation: a limit of learning, not
of representation.
**Design** (`pfn_pre.py`, `train_pre.py`, `gpu/jobs_e49.txt`, `e49_eval.py`, `e49_summ.py`; `train_sw.py`, `train_t2.py`,
`t2_world.py`, `e30_fmt.py`, `e30_eval.py`, `e37_eval.py`, `e40_eval.py`, `pfn2.py` and `world.py` unchanged; GPU scripts
`gpu/gpu_jobs_x.sh`, `gpu/launch_x.sh`, `gpu/retry_x.sh`).
- The variant "pre": x <- x + MLP(LN(x)) on every token after the input embedding and before block 1 (hidden 64, GELU; 4,256
  extra parameters, as E19). ctrl format, amendment 45's fixed 90,000-step schedule, seeds 0-11, in world 1 and in world 2 at
  D = 2 and D = 1 (N = 8); the same initialisation of every shared parameter and the same episodes as the E37 and E40 ctrl nets
  of each seed. 36 trainings on one on-demand L4 VM (8 at once; deadlines as amendment 45); estimated $1.9. Sparse checkpoints.
- Measures (`e49_eval.py`): world 1, e37_eval.py's (test KL; rom and type; block-2 labelled attention, xq; the sign-flip cost);
  world 2, e40_eval.py's (test KL, also by noise level; the sign-flip and negation costs).
- Acquired and r as amendment 56; r(pre/ctrl) over seeds acquired in both; c_sign pooled over acquired nets; 2,000 bootstrap
  resamples of seeds, with multiplicities.
**Predictions** (A-lin, architecture form).
- P198 (world 1): r(pre/ctrl) <= 0.6.
- P199 (world 2, D = 2): r(pre/ctrl) <= 0.6.
- P200 (the sign dependence falls): c_sign(pre) <= 0.5 x c_sign(ctrl) in world 1 and in world 2 at D = 2.
**Also reported (descriptive):** D = 1; stalls; A/B counts, rom, lab and xq (world 1); KL by noise level and negation costs (world
2); per-seed tables; K(pre) against E48's linear formats (sq; prod) and the folds (mag; canon) once E48 is evaluated.
**What the outcomes would mean.**
- P198-P200 hold: a per-token nonlinearity before the first pooling, an architecture change with the signed input, removes most
  of the departure and of the sign dependence in both worlds; with E48, the departure is predicted from the architecture's order
  of per-token computation and pooling.
- P198 or P199 fails while E48's linear formats gain: the architecture can represent the statistics, but training does not
  find the per-token computation; the departure is then a learning limit of this recipe (consistent with depth 3 in E38).
- P200 fails with P198-P199 holding: the KL falls but the nets still depend on the signs; the sign dependence is then not a
  by-product of computing the statistics after pooling.


## Amendment 58 (2026-10-30 12:05Z): E50, does block 1's pre-attention LayerNorm cause the big-miss departure? A LayerNorm ablation in three formats (architecture; GPU)

Written by the fifth agent before any E50 training. Known beforehand: E1-E49 (E48's write-up and review, Update 58; E49's
registered results, not yet written up or reviewed). In particular:
- E12-E13, E15: in every acquired net, a source with one big miss earns more weight than the same squared total spread out
  (large-range rom 0.133 in E37's ctrl nets); E24-E25 locate it in the source's own block-1 pooling.
- E20 (toy): LayerNorm followed by mean pooling undercounts a single big miss (capped per-token evidence); an oracle log-SS
  readout has no departure. E22: the toy's attention account fails in the real nets. The cap has not been tested in them.
- E48: r^2 in the reading slot (sq) kept ctrl's departure (rom 0.130; KL 1.81 x mag's); |r| in that slot (mag) 0.073; r^2 as an
  extra feature next to the signed reading (sgnsq) 0.087.
- E49 (pre, a per-token MLP before the first attention, whose input is LayerNormed): in world 1 it removed most of the sign
  dependence (sign share 0.017 against 0.200) and halved the KL on 7 paired acquired seeds (0.49), but kept the departure (rom
  0.126 over 8 acquired nets); 4 of 12 seeds stalled.
Plumbing (disclosed; not results):
- `pfn_noln.py` builds pfn2's PFN with blocks[0].ln1 replaced by the identity. A LayerNorm's initialisation is the constant
  (1, 0), so in CPU smoke runs (seed 0) every remaining parameter at step 0 equals the baseline's (E37 ctrl s0 for ctrl and mag,
  E48 sq s0 for sq), and only blocks.0.ln1.weight and .bias are absent; with that LayerNorm restored the variant computes exactly
  the baseline's function (`e50_plumb.py`);
- `train_noln.py` (train_sw.py with e48_fmt.py and only the model builder replaced) ran 3,000 constant-lr steps in each format
  (validation gaps at 3,000 steps: ctrl 0.183, mag 0.175, sq 0.146; slower than with the LayerNorm; not results;
  `e50_plumb.sh`); `e50_eval.py` ran on these nets; `e50_summ.py` ran on synthetic inputs and recovered the planted values, a
  planted stall and the planted decisions.
**Link to M2 (predicting the departure from the architecture).** The big-miss departure has been this program's lead since E12,
and it survives every format change (E48) and a per-token MLP before pooling (E49). E20's toy explains it through capped
per-token evidence: block 1's attention reads LayerNormed tokens, so a large error's contribution to the source's pooled summary
saturates, and a big miss counts for less than its square. If the departure disappears when only that LayerNorm is removed (and
the net is given r^2 linearly, so an uncapped pooled sum is exactly the statistic Bayes needs), it is predicted by one named
architectural component. If it persists, the cap is not its cause, and it is made elsewhere (attention weighting, the readout or
block 2).
**Design** (`pfn_noln.py`, `train_noln.py`, `gpu/jobs_e50.txt`, `e50_eval.py`, `e50_summ.py`; `train_sw.py`, `e48_fmt.py`,
`e48_eval_w1.py`, `e37_eval.py`, `e30_eval.py`, `pfn2.py` and `world.py` unchanged; GPU scripts `gpu/gpu_jobs_x.sh`,
`gpu/launch_x.sh`, `gpu/retry_x.sh`).
- The variant: block 1's pre-attention LayerNorm removed (queries, keys and values affine in each token's input embedding);
  block 1's MLP LayerNorm, block 2 and the final LayerNorm unchanged. World 1, formats ctrl, mag and sq, amendment 45's fixed
  90,000-step schedule, seeds 0-11; with the same initialisation of every remaining parameter and the same episodes as E37's ctrl
  and mag nets and E48's sq nets of each seed. 36 trainings on one on-demand L4 VM (8 at once; deadlines as amendment 45);
  estimated $2. Sparse checkpoints.
- Measures (`e50_eval.py`, e48_eval_w1.py's): test KL; rom (large and middle range) and the type; block-2 labelled attention and
  xq; the sign-flip cost for ctrl.
- Acquired gap < 0.10; r(a/b) = mean K_a / mean K_b over seeds acquired in both; rom compared as the mean over seeds acquired in
  both arms; 2,000 bootstrap resamples of seeds, with multiplicities.
**Predictions** (the cap causes the departure).
- P201 (with r^2 given, removing the cap removes the departure): rom_large(sq without LN) <= 0.5 x rom_large(sq with LN).
- P202 (once uncapped, the linear statistic beats the fold): r(sq without LN / mag without LN) <= 0.8.
- P203 (the cap causes the departure in the default format too): rom_large(ctrl without LN) <= 0.5 x rom_large(ctrl with LN).
**Also reported (descriptive):** KL with and without the LayerNorm per format; stalls; A/B counts; mag's rom without the
LayerNorm (|r|'s concave shape should keep part of the departure even uncapped); ctrl's sign share without the LayerNorm (in the
toy picture, LayerNorm is also a route by which an even function of the signed error reaches the pool); lab and xq.
**What the outcomes would mean.**
- P201 holds: given the statistic Bayes needs, the departure is the LayerNorm cap; with P203, the cap also causes it in the
  default format, and the departure is predicted from one architectural component.
- P201 fails: the cap is not what makes the departure even when r^2 is given; it arises in the pooling's weighting, the readout
  or block 2 (E22, E24).
- P202 holds: in E48 the cap hid the advantage of the linear statistic; P202 fails: |r| is as good as r^2 even uncapped.
- P203 fails while P201 holds: in the signed format the departure has another source as well, for example the even computation
  the signed format needs (E48).


## Amendment 59 (2026-10-30 12:22Z): E51, are the nets' sign-induced shifts shared across seeds, and is the sign-averaging gain an ensemble effect? (CPU, existing nets)

Written by the fifth agent before any E51 quantity is computed on the registered data, apart from the disclosed small-sample
plumbing run below. Known beforehand: E1-E49 with their reviews (Updates 58-59); E50 registered and training, no result. In
particular:
- E40 and E48's post hoc check (identical episodes, one estimator): at D = 1, averaging each ctrl net over 9 sign patterns lowers
  its KL from 0.0029 to 0.0019, below every trained sign-free or product-format net (0.0023-0.0029), while a single ctrl member
  (0.0029) is as good as a canon net (0.0029). At D = 2 and in world 1 the sign-free nets beat the sign-averaged ctrl nets.
- E6, E43: nets trained separately share part of their deviation from Bayes (correlation about 0.4-0.5 between unrelated nets of
  one type in world 1).
- E27-E28: in world 1 the sign dependence goes mostly with each source's overall error direction; no simple summary predicts it.
Plumbing (disclosed; not results):
- `e51_run.py` ran on the first 60 episodes of each setting into a scratch directory (`e51_plumb.sh`), and `e51_summ.py` ran on
  those outputs. These 60 episodes are part of the registered data. On them, with the registered statistics (noisy):
  - rho_h = 0.12 (world 1), 0.16 (D = 1), 0.26 (D = 2); c_sign 0.17, 0.51, 0.22;
  - c_seed for ctrl 0.50, 0.67, 0.29; for the invariant formats mag 0.39 (world 1), canon 0.57 and prod 0.59 (D = 1), canon 0.12
    and prod 0.24 (D = 2).
- The predictions below and their bars were written into `e51_summ.py` (file time 12:21:32Z) before that run (its first outputs
  12:21:54Z). A first plumbing run of world 1 failed on a file-name glob that also matched log files; it was fixed before the
  rerun.
- `e51_plumb.py` recovered planted cross-net correlations of 0, 0.3 and 0.7 (pair correlations 0.001, 0.300, 0.700).
**Link to M2 (why they deviate; what is predictable).** The measured sign dependence (c_sign, the gain from averaging a net over
sign patterns that Bayes ignores) has been this program's main measure of an invariance failure (E27-E49). E48 showed that it
does not predict the retraining gain; at D = 1 it is a gain that no trained invariant net reached. Two readings:
- (i) systematic: the nets share a sign-dependent bias, an inductive bias of the architecture that could be predicted from it;
- (ii) net-specific: each net's prediction varies with the signs like noise, so averaging over sign patterns acts as an ensemble;
  a net built to be invariant then has its own net-specific errors of similar size, which averaging over seeds can reduce but
  averaging over signs cannot.
Which reading holds decides whether the sign dependence belongs to the predictable departure or to the variation between
trainings, and it bears on why the input format helped in some settings and not others.
**Design** (`e51_run.py`, `e51_summ.py`; `e26_run.py`, `e27_data.py`, `e30_eval.py`, `e40_eval.py`, `e48_eval_t2.py`, `t2lin.py`
and `t2_world.py` unchanged). Existing nets only, nice 19:
- world 1: E27's first 3,000 bases, its flip_ind family (the base and 8 independent sign-flip patterns of the labelled errors);
  E37's acquired ctrl nets (11) and mag nets (12);
- world 2, D = 1 and D = 2: E40's 6,000 cost episodes and its 8 per-example flip patterns; E40's ctrl and canon nets and E48's
  prod nets (12 each).
- Per ctrl net: predictive means on every family member, and the family's mean KL and the KL reduction of its normalised
  geometric mean (`e26_run.set_stats`). Seed ensembles: the normalised geometric mean of 9 different nets' predictives on the base
  input (5 random 9-subsets, numpy seed 51), per format, with the same estimator (world 2: the per-episode grid of
  `e40_eval.family_grid` over all densities involved).
- Statistics (`e51_summ.py`): h_s = a net's predictive mean on a family member minus its mean over the 9 members; rho_h = the
  mean pairwise correlation of h across ctrl nets over all members and episodes; sys_h = the shared share of h's variance;
  rho_g = the pairwise correlation of the nets' sign-averaged deviations from the Bayes mean; c_sign = pooled family reduction /
  pooled family KL; c_seed(F) = the mean over subsets of the seed ensemble's reduction / its members' mean KL. 2,000 bootstrap
  resamples of nets.
**Predictions.**
- P204 (world 2, D = 1: the sign shifts are mostly net-specific): rho_h <= 0.2.
- P205 (world 2, D = 1: an ensemble effect): c_seed(canon) >= 0.75 x c_sign(ctrl).
**Also reported (descriptive):** rho_h, sys_h, rho_g, c_sign and c_seed in world 1 and at D = 2 and for every format; the member
and ensemble KLs.
**What the outcomes would mean.**
- P204 and P205 hold: at D = 1 the ctrl nets' sign dependence is mostly net-specific variation, and averaging over sign patterns
  acts as an ensemble over variants of one net, comparable with an ensemble of independently trained invariant nets. The measured
  sign cost then measures variation, not a shared bias, which fits the null format gain at D = 1 (E40, E48).
- P204 fails (shared shifts at D = 1): the nets share a sign bias that the invariant nets avoid, yet the invariant nets are no
  better; that contradiction would need another cost of the invariant formats.
- P205 fails: averaging over sign patterns gains more than an ensemble of invariant nets; the sign-dependent variation is then
  larger than the seed-to-seed variation of invariant nets.
- World 1 and D = 2 (descriptive): a high rho_h would mean the sign shifts there are shared across seeds, a systematic and
  potentially predictable part of the departure.


## Amendment 60 (2026-10-30 13:06Z): E52, does size shrink the shared part of the departure or the part that varies between trainings? (CPU, existing nets)

Written by the fifth agent before any E52 quantity is computed on the registered data. Known beforehand: E1-E51 with their
reviews (Updates 58-60); E50 registered and training, no result. In particular:
- E51 (world 1, width 32, the same data and estimator): 9-net ensembles reduce the mean single-net KL by 55% (ctrl, 0.0132 to
  0.0059) and 43% (mag, 0.0061 to 0.0034); the nets' deviations from the Bayes mean correlate 0.47 (ctrl) and 0.56 (mag).
- E38: mean test KL of ctrl nets 0.0127 (width 32), 0.0097 (64), 0.0072 (128), 0.0100 (depth 3); mag 0.0055, 0.0080 (three B
  nets), 0.0027, 0.0043; every E38 net acquired; A/B types among ctrl nets 4/7, 2/8, 4/6, 4/6.
Plumbing (disclosed; not results): `e52_run.py` ran on E27's bases 5,900-5,959, outside the registered bases (0-2,999), for all
four configurations into a scratch directory, and `e52_summ.py` ran on those outputs. Its last printed line, P207's decision on
those 60 other bases ("holds": false), was seen; nothing else from that run was inspected. The predictions and bars below were
written into `e52_summ.py` (file time 13:06:19Z) before that run (first output 13:06:25Z).
**Link to M2 (across sizes; what an architecture-level account can predict).** E51 found that about half of a single net's KL
varies between trainings and is removed by averaging nets, while the rest is shared. If size mainly removes the variation
between trainings and leaves the shared part, then the shared part is an inductive bias of the architecture family, which a
general account should predict, and the rest is a finite-size effect. If size shrinks both in proportion, the departure is a
capacity effect throughout.
**Design** (`e52_run.py`, `e52_summ.py`; `e26_run.py`, `e30_eval.py`, `e30_fmt.py` unchanged). Existing nets, world 1, fixed
90,000-step schedule, ctrl and mag formats: width 32 depth 2 (E37's acquired nets), width 64 and 128 depth 2 and width 32 depth 3
(E38's nets, 10 per cell). On E27's first 3,000 bases (base inputs) with E51's world-1 grid and estimator: each net's base
predictive; 9-net geometric ensembles over 5 random 9-subsets of the acquired nets (numpy seed 52). Statistics as in
`e52_summ.py`'s docstring: K1, K9, c_seed, rho_g, and the model-based shared and net-specific parts (descriptive). CPU, nice 19,
four processes.
**Predictions.**
- P206 (wider nets vary less between trainings): c_seed(width 128) <= c_seed(width 32) - 0.10, for ctrl and for mag.
- P207 (the shared part persists with width): K9(ctrl, width 128) >= 0.6 x K9(ctrl, width 32).
**Also reported (descriptive):** every statistic for width 64 and depth 3; rho_g by size; the model-based split.
**What the outcomes would mean.**
- P206 and P207 hold: size mainly removes the variation between trainings and leaves a shared departure, an inductive bias of
  this architecture family that persists across sizes (with E38, the signed format's cost is part of it).
- P206 fails and P207 fails: size shrinks the shared and the varying parts alike; the departure is a capacity effect.
- P206 fails and P207 holds: the shared part persists and the variation does not shrink either.
- P206 holds and P207 fails: size removes both, the varying part faster.


## Amendment 61 (2026-10-30 14:42Z): E53, is the big-miss departure made by the learned weighting of the source's pool? Uniform-pool variants and a constrained-Bayes test (architecture; trained on CPU)

Written by the sixth agent before any E53 training. Known beforehand: E1-E52 with their reviews (Updates 1-62). In particular:
- E12-E13, E15: in every acquired net a source with one big miss earns more weight than the same squared total spread out (E37's
  acquired ctrl nets: large-range rom 0.133; A-type 0.08-0.10, B-type 0.15-0.19). E24-E25 place it in the source's own block-1
  pooling at its query-question (summary) token, weights and values jointly. E22: replacing those weights by their within-source
  mean at evaluation lowers it in 46 of 48 nets and reverses it in 38 (off-distribution); E22's plumbing: 98.6% of that token's
  block-1 attention falls on its own labelled tokens, with within-source entropy 1.02 nats (uniform 1.39).
- E50: retrained without block 1's pre-attention LayerNorm, it persists in the ctrl and sq formats (0.160 against 0.127 on 6
  seeds; 0.124 against 0.130) and roughly halves with |r| (0.036 against 0.073). E48-E49: it persists through the formats and a
  per-token MLP before pooling. E20: in the single-source toy, a uniform mean of LayerNormed error tokens produces it (B type).
Prediction inputs computed before this amendment (properties of world 1, no net involved; `e53_cb.py`, `e53_cbkl.py`). A
"constrained-Bayes" reader is exact Bayes except that each source's labelled errors are seen only through a statistic T
(likelihood p(T | v) by simulation, 4 million draws per level); on E12's pairs (rom as E12) and on the 16,000 test episodes (KL
from Bayes by quadrature):
- T = SS: rom 0 and KL 0 (checks);
- T = mean |r|: rom_large 0.131, rom_middle 0.201; KL 0.00464 (se 0.00008);
- T = mean of r^2 / (1 + r^2 / 4) (a square capped at scale 2): 0.142, 0.073; KL 0.00060;
- T = mean of r^2 / (1 + r^2): 0.249, 0.240; KL 0.00430; T = max |r|: -0.455, -0.597 (NLL-based KL 0.0083).
That the capped square at scale 2 gives roughly the B-type nets' pattern at a small KL cost was noticed here; it is not tested by
this amendment.
Plumbing (disclosed; not results; `e53_plumb.py`, `e53_plumb.sh`):
- `pfn_uni.py` adds no parameter; the smoke runs' step-0 weights equal E37 ctrl s0's (uni, unif) and E50's no-LayerNorm mag s0 and
  sq s0's (unif_noln), every parameter; `block_fwd` without fixed rows equals pfn2's Block.forward exactly; the summary rows are
  exactly 1/4 on their own labelled tokens; in unif and unif_noln the query row has weight 0 on every labelled token in both
  blocks; at step 0, zeroing the labelled tokens' states after block 1 changes uni's output (0.80) and leaves unif's and
  unif_noln's unchanged (0); unif_noln's outputs on E12's two pair versions are identical in the sq format (6e-7) and differ in
  the mag format (0.02).
- 3,000-step CPU smoke runs of seed 0 in each arm (`--const 3000 --cooldown 0`; on CPU these are the first 3,000 steps of the
  registered seed-0 runs). Seen before this amendment: validation gaps at 3,000 steps 0.046 (uni), 0.046 (unif), 0.042
  (unif_noln mag), 0.030 (unif_noln sq); `e53_eval.py` on them: rom_large / rom_middle 0.262 / 0.241 (uni), 0.236 / 0.273 (unif),
  0.120 / 0.214 (unif_noln mag), 0 / 0 (unif_noln sq). The predictions, their bands and P211's form were written into
  `e53_summ.py` before the smoke runs started (14:38:35Z); the constant KCB_MABS there was then changed from a 0.0051 placeholder
  (the NLL-based estimate) to the quadrature value 0.00464 (the quadrature had been started at 14:28Z, before any smoke run).
- `e53_summ.py` on synthetic inputs recovered the planted values and decisions, with a planted stall.
**Link to M2 (predict the departure from the architecture).** The big-miss departure has survived every change at the input and
before the first pooling (E48-E50); the remaining architectural candidates are the learned weighting of the source's pool (E22,
E24) and the computation after it. This episode removes the learned weighting from the pool during training, with and without
the other routes from the labelled tokens to the output. If the departure goes, it is made by how the pool's attention learns to
weight a source's errors, and the account must explain that learning. If it stays in a net whose only route from a source's
errors is a uniform mean of LayerNormed tokens, the per-token features suffice (E20's account, in the real architecture). With a
uniform pool of linear features (no LayerNorm) the net sees exactly a known statistic of each source (mean |r| in the mag format,
SS / 4 in the sq format), so its departure and KL can be predicted before training by constrained Bayes. If they match, a net's
departure is predicted from the statistic its architecture pools plus a Bayes-optimal reader: a quantitative architecture-level
account in a controlled case, and a template for explaining the standard nets.
**Design** (`pfn_uni.py`, `train_uni.py`, `e53_eval.py`, `e53_summ.py`, `e53_q.txt`; `train_sw.py`, `e48_fmt.py`, `e30_fmt.py`,
`pfn_noln.py`, `pfn_pre.py`, `e48_eval_w1.py`, `e37_eval.py`, `e30_eval.py`, `pfn2.py` and `world.py` unchanged).
- Variants (no parameter added; with the same seed every parameter starts as in the baseline net, and the episodes are the
  baseline's):
  - uni: in block 1, each summary token (4, 9, 14) attends with fixed weights 1/4 to its own source's four labelled tokens and to
    nothing else, in both heads; values, every other attention row and block 2 learned. ctrl format, seeds 0-11.
  - unif: uni, and the query token may not attend to any labelled token in either block, so every route from a labelled token to
    the output passes through the uniform pool. ctrl format, seeds 0-11.
  - unif_noln: unif with block 1's pre-attention LayerNorm removed (E50's variant); the summary then sees exactly mean |r| (mag)
    or SS / 4 (sq) of its source plus error-free terms. mag and sq formats, seeds 0-7.
- World 1, amendment 45's fixed 90,000-step schedule, train_sw.py's recipe, sparse checkpoints. 40 trainings on General's CPU
  (nice 19, ionice -c3, at most 4 at once; queue `e53_q.txt`), instead of a GPU, to save the lane's GPU budget; the comparison
  nets were trained on L4 GPUs, so arithmetic differs at the level of float rounding (disclosed).
- Measures (`e53_eval.py`, e48_eval_w1.py's): test KL; rom large and middle and the type; block-2 query attention to the labelled
  tokens (lab; 0 in unif by construction) and xq; the sign-flip cost (ctrl format).
- Comparisons, same seeds: E37's ctrl and mag nets, E50's no-LayerNorm mag and sq nets, E48's sq nets. Acquired: gap < 0.10.
  Paired seeds acquired in both; 2,000 bootstrap resamples of seeds, with multiplicities.
**Predictions** (the learned weighting makes the departure; a net reads its pooled statistic like Bayes).
- P208: rom_large(uni) <= 0.5 x rom_large(ctrl), paired seeds.
- P209: rom_large(unif) <= 0.5 x rom_large(ctrl), paired seeds.
- P210 (constrained Bayes): unif_noln in the mag format, means over acquired nets: rom_large in [0.091, 0.171] and rom_middle in
  [0.141, 0.261] (the reader of mean |r|: 0.131 and 0.201, +-0.04 and +-0.06).
- P211 (additivity): the mean paired KL difference, unif_noln mag minus unif_noln sq, is within [0.7, 1.3] x 0.00464 (the
  constrained-Bayes KL of mean |r|), i.e. [0.00325, 0.00603].
**Also reported (descriptive):** KL, stalls, types, lab, xq and the sign share in every arm; unif_noln sq's KL (the KL left with an
exact per-source statistic; its rom is 0 by construction) against mag's and sq_noln's; unif_noln mag's KL against its lower bound
0.00464; ratios against ctrl, mag and the no-LayerNorm nets; validation curves.
**What the outcomes would mean.**
- P208 and P209 hold: the learned weighting of the pool makes the departure; a uniform mean of LayerNormed tokens does not make it
  in the real architecture (against E20's toy). The account must then explain why training learns that weighting.
- P209 fails (unif keeps more than half): a uniform mean of LayerNormed tokens suffices for the departure, as in E20's toy; with
  E50 (it persists without that LayerNorm under learned weighting) there would be two sufficient routes.
- P208 fails and P209 holds: with only the pool fixed, the departure re-forms through the learned routes (block 2's or the query's
  reading of the labelled tokens); closing them removes it.
- P210 holds: in the controlled architecture, the trained nets' departure is the constrained-Bayes departure of the statistic
  they pool; P210 fails: the reader adds its own departure (e.g. a miscalibrated reading of the statistic).
- P211 holds: the KL separates additively into the statistic's information loss and a downstream part; fails: the downstream
  part depends on the statistic.


## Amendment 62 (2026-10-30 14:55Z): E54, does world 2 show the big-miss fingerprint? Matched contexts with the same sufficient statistics and a concentrated or a spread residual (CPU, existing nets)

Written by the sixth agent before any E54 quantity is computed on the registered data. Known beforehand: E1-E52 with their
reviews; E53 registered (amendment 61) and training, no result. In particular:
- World 1: every acquired net gives a source with one big miss more weight than a source with the same SS spread out (E37's ctrl
  nets: large-range rom 0.133); it persisted through every format and architecture change tried (E48-E50). Amendment 61's
  constrained-Bayes values: an exact reader of mean |r| gives rom_large 0.131; a reader of a square capped at scale 2 gives
  0.142 / 0.073 at a KL cost of 0.0006 nats.
- World 2 (E40, E48, E49): in-context linear regression with an unknown noise level (D = 1, 2; N = 8 examples). The nets depend
  on per-example signs (about a third of their KL); folding the signs (canon) or supplying the products (prod, sgnprod) halves the
  KL at D = 2; a per-token MLP before the first attention (pre) lowers it by about a quarter. No test of how the residual is spread
  over the examples has been made in world 2.
Plumbing (disclosed; not results; no effect estimate on any net was computed or seen):
- `e54_data.py` on seed 54,900 (300 bases per D, outside the registered data): Bayes identical within pairs to 3e-14; X^T y and
  y.y conserved to 6e-14; median largest residual share 0.92 (big) and 0.22 (spread) at D = 1, 0.80 and 0.30 at D = 2.
- `e54_run.py` ran on two nets (E40 D = 1 canon seeds 0 and 1) on those pairs, printing only array shapes, finiteness and the
  grid mass (>= 0.99999); its loader reproduces the recorded test KL of one net of each kind exactly (E40 D = 1 ctrl, D = 2 canon;
  E48 D = 1 prod, D = 2 sgnprod; E49 D = 1 and D = 2 pre).
- `e54_summ.py` on synthetic per-net results (`e54_synth.py`) recovered the planted values and decisions.
**Link to M2 (across tasks; the inductive bias).** The big-miss departure is the program's most robust fingerprint in world 1.
World 2's nets see raw (x, y) examples and no sources, and the evidence about the noise level is the residual sum of squares
(RSS), a sum of squared per-example deviations, as SS is in world 1. If these nets also treat a residual concentrated on one
example as less than the same RSS spread out (a narrower predictive, and a mean that moves as if the outlier were discounted),
the fingerprint is a property of how these transformers pool per-example evidence in context, not of the per-source task, and
one account (per-example evidence that saturates before or within pooling, read by a near-Bayes reader) could cover both tasks.
If it is absent or reversed, the world-1 departure is tied to that task's structure.
**Design** (`e54_data.py`, `e54_run.py`, `e54_summ.py`; `t2_world.py`, `t2lin.py`, `e40_eval.py`, `e26_run.py`, `pfn2.py` and
`pfn_pre.py` unchanged; queue `dynq.sh`).
- Pairs: for D = 1 and D = 2, 4,000 base episodes (t2_world.sample, N = 8; seeds 54,001 and 54,002). Least squares w_hat,
  residual e (X^T e = 0), RSS = e.e, P the projection off the column space of X. big: residual s sqrt(RSS) P d_j / |P d_j| (j a
  random example, s a random sign), most of RSS on example j; spread: sqrt(RSS) P sig / |P sig| (sig random signs). Both versions
  keep X^T X, X^T y and y.y, so Bayes is identical. Calibration versions: the spread version with its residual scaled by 0.6 and
  0.8 (RSS x 0.36 and 0.64).
- Nets: every world-2 net, best-validation model.pt: E40's ctrl and canon, E48's prod and sgnprod, E49's pre (ctrl format), at
  D = 1 and D = 2, 12 seeds each (120 nets; all acquired).
- Per net: dls = mean (log-sd_big - log-sd_spread) of the predictive (mixture moments); lam = the coefficient on
  d = x*.(w_hat_{-j} - w_hat) (the shift that leaving out the outlier would cause, on the big version) in a least-squares fit of
  (mean_big - mean_spread) on [d, Bayes mean, 1]; dkl = mean (KL_big - KL_spread) (per-episode grids as e40_eval.family_grid);
  u (descriptive) = the share of RSS the net behaves as if it ignored, 1 - exp(2 dls / b), with b the slope of the spread
  versions' mean log-sd on log f.
- Arm means over nets; 2,000 bootstrap resamples of nets, with multiplicities.
**Predictions** (the departure is a general bias toward under-counting a concentrated deviation).
- P212: ctrl nets at D = 1 and at D = 2: mean dls < 0, and dls < 0 in >= 10 of 12 nets.
- P213: ctrl nets at D = 1 and at D = 2: mean lam > 0, and lam > 0 in >= 10 of 12 nets.
- P214: mean dls < 0 in >= 8 of the 10 arms (five kinds of net, two D).
**Also reported (descriptive):** dls by true noise level; u; dkl; kappa (the coefficient on the Bayes mean, a change in
shrinkage); per-net values.
**What the outcomes would mean.**
- P212-P214 hold: both tasks share the fingerprint, across every format and variant tried in world 2: these transformers
  under-count a large per-example deviation in what they pool, as a robust estimator does.
- P212 holds and P213 fails: the noise-level estimate under-counts the outlier but the estimate of the line does not discount it;
  the bias is in the scale statistic only.
- P212 fails (dls >= 0, or mixed signs across nets): world-2 nets do not under-count a concentrated residual; the world-1
  departure is tied to the per-source pooling of directly visible errors.
- P214 fails while P212 holds: in world 2 the fingerprint depends on the input format or the architecture variant.


## Amendment 63 (2026-10-30 15:44Z): E55, does block 1's LayerNorm cap scale, read from a net's input embedding, predict its big-miss departure? (CPU, weights and existing evaluations)

Written by the sixth agent before E55's geometry is computed for any confirmatory net and before any E53 net of seed 1 or later
is evaluated. Known beforehand: E1-E54 with their reviews (E54: Update 63); E53 registered (amendment 61) and training; E53's
seed-0 evaluations (uni 0.180 / 0.100, unif 0.177 / 0.106, unif_noln mag 0.138 / 0.203, unif_noln sq 0 / 0 for rom large /
middle; KL 0.0043, 0.0041, 0.0084, 0.0034) were printed by the pipeline and seen.
**The idea (derived before any of the data below).** With LayerNorm before the first attention, a labelled token's centred
embedding is a_i + r_i u~ (u~ the centred o column, a_i the rest of the token at r_i = 0), and LayerNorm divides it by
n_i = |a_i + r_i u~| = sqrt(A_i^2 + 2 r_i A_i U rho_i + r_i^2 U^2). A uniform mean of the tokens' values therefore carries an even
per-token feature 1 / n_i, which up to an affine map is h_c(r) = 1 - (1 + r^2 / c^2)^(-1/2) with cap scale c = A / U when
rho = 0 and A is constant: quadratic in r for small errors, saturating for large ones. A Bayes reader of the source's mean of
h_c (constrained Bayes, amendment 61) then shows a big-miss departure whose size is fixed by c alone (`e55_cbtable.py`, a
property of world 1, computed at 15:41:52-15:42:01Z: rom_large / rom_middle 0.211 / 0.176 at c = 1, 0.165 / 0.102 at 1.4,
0.107 / 0.053 at 2, 0.056 / 0.025 at 3, 0.033 / 0.014 at 4, 0.022 / 0.009 at 5).
**Exploration (disclosed; not results; on nets that the confirmatory tests below exclude).** `lngeom.py` (geometry; it
reproduces the LayerNorm's norm exactly) on E27's first 500 bases:
- at initialisation (seed 0): c 2.09, A's coefficient of variation 0.27, median |rho| 0.35;
- E37's acquired ctrl nets: c 3.56-4.41 except seed 8 (1.62, the worst B net); Pearson(c, rom_large) -0.74 over 11 nets; mag nets:
  c 4.33-4.91 except seeds 3 (1.69, the B net) and 11 (3.51); Pearson -0.95 over 12 nets. In these nets with learned pooling the
  measured departure exceeds the constrained-Bayes value at their c (ctrl seed 0: c 4.41, rom_large 0.086 against about 0.028);
- E53's seed-0 uniform-pool nets: c 1.44 (uni) and 1.39 (unif), A's coefficient of variation 0.01, |rho| 0.00-0.01; their
  measured rom (0.180 / 0.100, 0.177 / 0.106) against the constrained-Bayes value at c = 1.4 (0.165 / 0.102).
Earlier, for other purposes, this agent printed the arm means of rom for E30's and E38's nets (E30 fix 0.149; E38 width 64 ctrl
0.122, mag 0.079; width 128 ctrl 0.064, mag -0.002; depth 3 ctrl 0.089, mag 0.040); no c of any of those nets was computed.
Plumbing (disclosed): `e55_summ.py` on a synthetic tree (`e55_synth.py`) recovered planted decisions; `e55_run.py` has not been run.
**Link to M2 (predict the departure from the architecture).** If a single number of the first layer's geometry, the cap scale
that LayerNorm gives the error direction, predicts the departure's size quantitatively where the pool is uniform (through
constrained Bayes, with no behavioural fit) and orders it where the pool is learned, the departure is predicted from the
architecture and its trained weights, by a mechanism (LayerNorm's normalisation makes per-token evidence saturate). With the
width and training-time parts (P217, P218), the same number would account for how the departure scales with width and decays
during training.
**Design** (`lngeom.py`, `e55_cbtable.py`, `e55_run.py`, `e55_summ.py`, `e55_synth.py`; `e48_fmt.py`, `e53_cb.py` unchanged).
- c of a net: the median over the labelled tokens of E27's first 500 bases, built in the net's own format (for mag and sq, "r" is
  the o column minus the y column, |r| or r^2), of A_i / U, from its model.pt's inp.weight and inp.bias; no behaviour is used.
- Predicted rom of a net: linear interpolation of the table at its c (19 grid points from 0.8 to 6; clamped and flagged outside).
- Measured rom and acquisition (gap < 0.10): the nets' registered evaluations (results/<episode>/<tag>.npz); for E53, e53_eval.py.
- Cohorts. Exploration (reported separately): E37 ctrl, E37 mag. Confirmatory, learned pooling: E38 (width 64, width 128, depth 3;
  ctrl and mag), E30 fix, E48 sq, E43 (br0, br2k, br3k, br5k, br10k, br20k, newinit), E44 (by swapped group: emb, b1att, b1mlp,
  b2att, b2mlp, head), E46 (eps 0.003, 0.03, 0.3). Uniform pool: E53 uni and unif (seeds 1-11; seed 0 explored). E30's ctrl and mag
  nets are excluded (E37's nets reproduce them).
- Training dynamics: c at every checkpoint of E37's pure ctrl and mag runs (26 checkpoints, 0-90,000 steps), against E37's
  trajectories of rom_large at the same steps (results/e37_traj/).
**Predictions.**
- P215 (uniform pool, quantitative): for E53's uni and for its unif nets (acquired, seeds 1-11), the mean measured rom_large and
  rom_middle are each within 0.03 of the mean constrained-Bayes prediction at the nets' c.
- P216 (learned pooling, ordering): in the confirmatory cohorts with >= 8 acquired nets, the within-cohort Spearman correlation of
  c with rom_large is negative in >= 80% of the cohorts, and its median is <= -0.4.
- P217 (width): the median c of E38's width-128 nets is larger than that of E37's width-32 nets, for ctrl and for mag.
- P218 (training): in E37's acquired pure ctrl and mag runs, the Spearman correlation of c with rom_large over the checkpoints at
  >= 5,000 steps is negative in >= 80% of the runs.
**Also reported (descriptive):** c, A's coefficient of variation and |rho| by cohort; measured against constrained-Bayes rom in
every cohort (where the pool is learned, the excess); c at 5,000 and 90,000 steps.
**What the outcomes would mean.**
- P215 holds: where the pool is uniform, the departure is the constrained-Bayes departure of LayerNorm's saturating per-token
  feature at the cap scale read from the weights, a quantitative prediction from the architecture.
- P216 holds: where the pool is learned, the same geometric number orders the departure across nets; the learned weighting then
  modifies (in E37's nets, adds to) a cap-set baseline. P216 fails: with learned pooling the cap scale is not what sets the
  departure's size across nets.
- P217 and P218 hold: the departure shrinks with width and during training as the learned cap scale grows; fail: the cap scale does
  not track those changes.


## Amendment 64 (2026-10-30 16:20Z): E56, does a shared input column limit the learned cap scale? Decoupling the summary readings from the labelled errors (architecture; trained on CPU)

Written by the sixth agent before any E56 training. Known beforehand: E1-E54 with their reviews; E53 (amendment 61) and E55
(amendment 63) registered and running, with these results seen: E53's seed-0 nets (amendment 63); E53's unif seeds 0-1 on average
rom_large / rom_middle 0.171 / 0.107 against the constrained-Bayes value at their c, 0.166 / 0.104 (seen in this amendment's
plumbing run, which reads E53's results); E55's P216-P218 on the confirmatory cohorts (results/e55.json, first run): c orders the
departure within cohorts (21 of 23 negative, median Spearman -0.76), width 128 has a larger median c than width 32, and within E37's
runs c rises while the departure falls (22 of 23 runs; after a disclosed correction of NaN handling); the cap's constrained-Bayes KL
cost (`e55_cbkl.py`, 4,000 test episodes): 0.0022 nats at c = 1, 0.0009 at 1.4, 0.0003 at 2, 0.00008 at 3, 0.00003 at 4.
**The idea.** In the ctrl format one input column (the o column) carries both the labelled tokens' values (o = y + r) and the
summary tokens' own readings o*. Its input weight u sets both the labelled errors' LayerNorm cap scale c = A / U (a small U,
i.e. a large c, means less saturation and, by constrained Bayes, a smaller departure) and how strongly each summary's reading
enters its token (relative to the codes and flags in the same LayerNormed vector, from which block 2 must read the reading
precisely). If that shared weight is what holds c down (about 1.4 with a uniform pool and 3.8 with a learned pool), giving the
readings their own column frees U, and the nets should learn a larger c and a smaller departure.
Plumbing (disclosed; not results; `e56_plumb.py`, `/tmp/s6/e56_smoke.sh`):
- `e56_fmt.py`'s sep format: labelled and query tokens equal ctrl's (extra column 0); on summary tokens the o column is 0 and the
  extra column (index 36) holds the reading; checked on random episodes. A sep net whose column-36 weights are copied from column 32
  computes the ctrl net's function (max difference 5e-7, learned and uniform pool).
- 3,000-step CPU smoke runs of seed 0 with `--pair_init` (on CPU, the first 3,000 steps of the registered seed-0 runs): every
  shared parameter equals E37 ctrl s0's (sep) and E53 unif s0's (unif sep) step 0, and inp.weight's first 36 columns equal theirs.
  Seen: validation gaps at 3,000 steps 0.187 (sep, not yet acquired) and 0.047 (unif sep); `e56_eval.py` on them: rom_large /
  rom_middle 0.046 / 0.015 (sep) and 0.198 / 0.282 (unif sep); c 0.76 (unif sep at 3,000 steps).
- `e56_summ.py` ran end to end on these smoke nets with the real comparison nets.
**Link to M2 (predict the departure from the architecture, and why training does not remove it).** E55's first results tie the
departure to one learned geometric number, the LayerNorm cap scale, which rises during training as the departure decays and is
larger in wider nets. This episode tests an architectural reason why it stops rising: weight sharing between the labelled values
and the readings in the input embedding. If decoupling raises c and lowers the departure (and, with a uniform pool, by the amount
constrained Bayes predicts at the new c), the departure is explained as a consequence of LayerNorm plus a shared input column,
and an architecture without that sharing is predicted to depart less.
**Design** (`e56_fmt.py`, `train_e56.py`, `e56_eval.py`, `e56_summ.py`, `e56_plumb.py`; `train_sw.py`, `pfn_uni.py`, `e48_fmt.py`,
`lngeom.py`, `e37_eval.py`, `e30_eval.py`, `e15_run.py`, `pfn2.py`, `world.py` unchanged).
- Arms (world 1, amendment 45's fixed 90,000-step schedule, train_sw.py's recipe with `--pair_init`, sparse checkpoints, seeds 0-11,
  the same episodes as the ctrl and unif nets of each seed):
  - sep: the baseline architecture (learned pool) in the sep format; every shared parameter starts as in E37's ctrl net of the seed;
  - unif sep: E53's unif variant in the sep format; every shared parameter starts as in E53's unif net (= E37's ctrl net) of the seed.
  The extra input column is drawn U(-1/6, 1/6) from torch.Generator(31337 + seed). 24 trainings on General's CPU through the
  priority queue after E53's trainings (nice 19, at most 4 heavy processes).
- Measures: `e56_eval.py` (e53_eval.py's, with the effective weights taken on the reading column, 36 for sep); c of every net by
  lngeom.py exactly as e55_run.py (labelled tokens; the o column now carries only labelled values); the readings column's norm U2.
- Comparisons: E37's ctrl nets and E53's unif nets of the same seeds (their c from results/e55_geom.json). Acquired gap < 0.10;
  paired seeds acquired in both; 2,000 bootstrap resamples of seeds, with multiplicities.
**Predictions.**
- P219 (learned pool): the median over paired seeds of c(sep) / c(ctrl) is >= 1.25, and mean rom_large(sep) <= 0.7 x mean
  rom_large(ctrl).
- P220 (uniform pool): the median of c(unif sep) / c(unif) is >= 1.5, and mean rom_large(unif sep) <= 0.6 x mean rom_large(unif).
- P221 (constrained Bayes at the new cap scale): for unif sep, the mean measured rom_large and rom_middle are each within 0.03 of
  the mean constrained-Bayes prediction at the nets' c (table of amendment 63).
**Also reported (descriptive):** KL ratios against ctrl and unif; A / B counts; U and U2; sign shares; lab; acquisition.
**What the outcomes would mean.**
- P219 and P220 hold: the shared input column holds the cap scale down; decoupling the readings lets training un-cap the labelled
  errors and shrinks the departure. With P221, the size of the change is the constrained-Bayes prediction at the new c.
- Both fail on c (no rise): the cap scale is set by something other than the shared column (for example the amplitude the reader
  needs from the even feature), and the account must look there.
- c rises but the departure does not fall (learned pool): the learned weighting's part of the departure does not scale with c.


## Amendment 65 (2026-10-30 16:24Z): E57, is the A/B solution type a cap-scale type, and when does the cap scale separate the types? (CPU, checkpoint weights)

Written by the sixth agent before E57's cap scales are computed. Known beforehand: E1-E54 with their reviews; E47 (Update 57):
the eventual type is not separated by block-2 labelled attention, block-1 cross-source attention (xq), the sign-cost share or KL
at initialisation, weakly at 3,000 steps, and by xq and the sign share with AUC 0.97 at 10,000 steps; E44: block 1's attention
weights transfer the type most often, the input embedding rarely. From E55's registered computation (results/e55.json): at the end
of training, over the 205 acquired width-32 ctrl nets of E37, E43, E44 and E46 (E47's lineages), B nets have a lower cap scale
(median 3.70 against 4.20; P(c_B < c_A) = 0.94, unclustered), and in every other size and format the B nets' median c is below the
A nets'; E37's ctrl runs' c at 6,000 steps (printed in E55's first run, before this amendment) did not separate the types (A
1.51-1.81, B 1.27-1.81).
**Link to M2 (how they learn; the inductive bias).** E55 ties the size of the big-miss departure to the learned LayerNorm cap
scale, and the end-of-training association above ties the program's two solution types (E6-E47) to it as well: B nets keep a
smaller cap scale and a larger departure. Whether the cap scale separates the types as early as the routing markers (E47) or only
later decides whether the cap-scale difference is part of the type's origin or a downstream consequence of the routing that the
type adopts first; that ordering is what an account of how the departure is learned needs.
Plumbing (disclosed): `e57_summ.py` on synthetic cap scales with the real final types reproduced E47's per-step lineage counts
(149 at step 0, 193 at 10,000, 215 from 40,000) and the planted AUCs.
**Design** (`e57_run.py`, `e57_summ.py`; `lngeom.py`, `e47_summ.py`, `e48_fmt.py` unchanged). c at steps 0, 2,000, 3,000, 5,000,
10,000, 20,000, 40,000, 60,000, 80,000 and 90,000 for every E47 lineage (checkpoint weights only), as e55_run.py. Lineages, final
types, clusters and the duplicate rule exactly as E47. AUC_c(step) = P(c of a final-A lineage > c of a final-B lineage), with
2,000 cluster-bootstrap resamples; also without E37's lineages.
**Predictions** (the types separate first in routing; the cap scale follows).
- P222: AUC_c at step 0 is within [0.4, 0.6].
- P223: AUC_c at 10,000 steps is < 0.75 (when E47's xq separates the types at 0.97).
- P224: AUC_c at 40,000 steps is >= 0.80.
**Also reported (descriptive):** AUC_c at every step, with and without E37; median c by type and step; E47's marker AUCs at the
same steps.
**What the outcomes would mean.**
- P222-P224 hold: the cap scale does not mark the type at initialisation or when the routing markers already do, and separates it
  later in training: the B type's smaller cap scale (and with it part of its larger departure) develops after the routing choice,
  as a consequence of it, not as its origin.
- P223 fails: the cap scale separates the types as early as the routing markers; cap and routing diverge together.
- P224 fails: the end-of-training association arises only in the last part of training (the cooldown).
- P222 fails: the input embedding's initial geometry already carries the type.


## Amendment 66 (2026-10-30 16:53Z): E58, does setting the LayerNorm cap scale by design set the departure? A frozen designed input embedding (causal check; GPU, about $1.6)

Written by the sixth agent before any E58 training, in response to the coordinator's note (16:47Z): E55 and E57 tie the cap scale
to the departure only by association (across nets and over training), and E50 found that the departure survives removing block
1's LayerNorm (except that it roughly halves with |r|). Known beforehand: E1-E54 and E57 with their reviews; E53 (seeds 0-2 seen:
uni and unif rom_large 0.154-0.180, rom_middle 0.100-0.107; unif_noln mag 0.132-0.140 / 0.203-0.207), E55 (first run: P216-P218
hold; E53's unif seeds 0-1 average 0.171 / 0.107 against the constrained-Bayes value at their c, 0.166 / 0.104) and E56 registered.
**Why a trained-net intervention is not used.** In a trained net, rescaling the error direction of the input embedding (u -> k u)
is the same operation as rescaling every labelled error by k, so it moves the pairs to another SS range instead of changing only
the cap; and changing only the normalisation (the denominator) leaves the trained reader miscalibrated for the new statistic.
Instead the cap scale is set by design and everything downstream is trained with it.
**The tension with E50 (what this tests).** Under the account, a uniform pool has one nonlinearity that can turn signed errors
into a reliability statistic, LayerNorm's saturating feature, so there the cap scale should set the departure as constrained Bayes
predicts; a learned pool has a second one, the attention weighting, which E50 shows can make a departure without LayerNorm. So
the account predicts that the cap scale is causal in both but sufficient for the departure's size only with a uniform pool.
Plumbing (disclosed; not results; `e58_emb.py`, `e58_plumb.py`, `/tmp/e58_smoke`, `/tmp/e58_synth`):
- `e58_emb.py` takes E53's unif seed-0 net's trained input embedding (ctrl format; c 1.39, A's coefficient of variation 0.007,
  |rho| 0.005), gives the summary readings their own column (sep format, e56_fmt.py) with the reference's o-column weights, and
  scales only the labelled-error direction: column 32 (labelled y + r) by k = c_ref / c_target, column 33 (y) plus (1 - k) times
  the old column 32, so a_i and the summary and query tokens' embeddings are unchanged. lngeom.py on the designed embeddings: c =
  1.000, 2.000, 4.000 exactly (A's coefficient of variation 0.007, |rho| 0.005).
- `train_frz.py`: 600-step CPU runs (seed 0; unif at c = 2, learned pool at c = 4): at step 0 the embedding equals the design and
  every other parameter equals E53 unif s0's / E37 ctrl s0's step 0; after 600 steps the embedding is unchanged and the other
  weights moved. `e56_eval.py` and `e58_summ.py` ran end to end on those nets (not acquired at 600 steps); `e58_summ.py` recovered
  planted values and decisions on synthetic results.
**Link to M2 (predict the departure from the architecture; cause or correlate).** If setting the cap scale by design moves the
departure of uniform-pool nets along constrained Bayes's curve (a sixfold change in rom_large between c = 1 and 4), the cap scale
is a cause of the departure there, and its size is predicted from the architecture and one geometric number. If it also moves the
learned-pool nets' departure in the same direction, the cap is a cause there too, alongside the learned weighting (E50's route).
**Design** (`e58_emb.py`, `train_frz.py`, `gpu/jobs_e58.txt`, `e58_summ.py`, `e58_plumb.py`; `e56_eval.py`, `e56_fmt.py`, `e48_fmt.py`,
`train_sw.py`, `pfn_uni.py`, `pfn_noln.py`, `pfn_pre.py`, `lngeom.py`, `pfn2.py`, `world.py` unchanged; GPU scripts `gpu/gpu_jobs_x.sh`,
`gpu/launch_x.sh`, `gpu/retry_x.sh`).
- Arms (world 1, sep format, amendment 45's fixed 90,000-step schedule, train_sw.py's recipe, `--pair_init`, sparse checkpoints, seeds
  0-5): unif (E53's uniform factorised pool) with the input embedding frozen at c = 1, 2 and 4; the baseline learned pool frozen at
  c = 1 and 4. Every other parameter starts as in E53's unif / E37's ctrl net of the seed, with the same episodes. 30 trainings on one
  on-demand L4 (8 at once), estimated $1.6; lane GPU total about $33.6 of $40.
- Measures: `e56_eval.py` (test KL; rom on the readings column; type; lab, xq; sign-flip cost); c re-measured with lngeom.py.
- Constrained-Bayes values at the designed c: the table of amendment 63 (results/e55_cbtable.json): c = 1: 0.211 / 0.176; c = 2:
  0.107 / 0.053; c = 4: 0.033 / 0.014 (rom_large / rom_middle).
- Acquired gap < 0.10; 2,000 bootstrap resamples of seeds, with multiplicities.
**Predictions.**
- P225 (uniform pool, quantitative and causal): at each designed c, the mean rom_large and the mean rom_middle are each within 0.03 of
  the constrained-Bayes values at that c, and the mean rom_large falls from c = 1 to 2 to 4.
- P226 (learned pool, direction): rom_large(c = 1) > rom_large(c = 4) in >= 5 of the 6 seeds (acquired in both arms), and the mean
  difference is >= 0.05.
**Also reported (descriptive):** KL by arm (constrained Bayes predicts the cap's KL cost to fall from about 0.0022 nats at c = 1 to
0.0003 at 2 and 0.00003 at 4); stalls; A / B counts; sign shares; lab.
**What the outcomes would mean.**
- P225 holds: with a uniform pool, the cap scale causally sets the departure, by the amount constrained Bayes predicts; with E53's
  unif_noln arms (no LayerNorm: the departure is that of the exact statistic given, 0 for r^2), LayerNorm's saturation is the
  departure's cause in uniform-pool nets.
- P225 fails on the values but the departure falls with c: the cap is causal, but the reader is not near-optimal for every c.
- P225 fails with no monotone fall: the E55 match was not causal.
- P226 holds: in learned-pool nets the cap also causes part of the departure; with E50, the learned weighting is a second,
  independent source, which resolves the apparent tension. P226 fails: with a learned pool the cap scale does not drive the
  departure, and E55's ordering there is a correlate (for example of how the learned weighting develops).


## Amendment 67 (2026-10-30 17:39Z): E59, why are uniform-pool nets closer to Bayes? Identity-code dependence and run-to-run variation with and without the learned pool (CPU, E53's nets)

Written by the sixth agent before any E59 quantity is computed on a trained net (code-family costs and seed ensembles of E53's and
E37's nets have never been computed). Known beforehand: E1-E54 and E57 with their reviews; E53's seeds 0-2 (seen in its evaluation
log): uni and unif test KL 0.0041-0.0051 against E37 ctrl's mean 0.0127, sign-flip cost shares 0.009-0.034 (ctrl 0.200), unif_noln
sq KL 0.0028-0.0034 and unif_noln mag 0.0077-0.0087; E55's first run; E58 registered and training. Earlier results used here: E29
and E30 (the ctrl nets depend on the arbitrary source codes: averaging over code redraws removes about a fifth of their KL), E51-E52
(about half of a ctrl net's KL is removed by averaging 9 nets; c_seed 0.56 for E37's ctrl nets at width 32).
**Link to M2 (why they deviate; the inductive bias).** E53's early seeds suggest that fixing a source's pool to uniform weights
brings the nets about three times closer to Bayes and nearly removes their sign dependence. If the same change also removes most of
their dependence on the arbitrary identity codes and most of the variation between trainings, then the bulk of a standard net's
distance from Bayes here is made by the learned, content-based weighting of the pool (which must find a source's tokens by their
codes and can pick up the signs), not by the departure itself (whose constrained-Bayes cost is about 0.001 nats or less). That
would locate the main inductive bias in one mechanism and say what an architecture closer to Bayes needs.
Plumbing (disclosed): `e59_run.py` and `e59_summ.py` ran end to end on untrained (step-0) weights of nine E37 ctrl seeds in a
scratch tree (/tmp/e59_plumb, 60 bases); no trained net's quantity was computed.
**Design** (`e59_run.py`, `e59_summ.py`; `e26_run.py`, `e29_data.py`, `e30_eval.py`, `e48_fmt.py`, `pfn_uni.py` unchanged). Arms:
E37's acquired ctrl nets (11), E53's uni and unif (12 each), unif_noln mag and sq (8 each). Per net, with E51-E52's estimator on
E27's first 3,000 bases: the base KL; the identity-code family (E29's SID family, all 6,000 of its bases: the base and 8 redraws of
the three source codes) as e30_eval.family_cost: its mean KL and the KL reduction of averaging over it; the sign family cost from
the registered evaluations. Seed ensembles: normalised geometric means of K nets of an arm on the base inputs, 5 random K-subsets
(numpy seed 59), K = 7 for every arm and K = 9 for ctrl, uni and unif. Shares: code share = sum of reductions / sum of family KLs
over the arm's nets; c_seed(K) = the mean over subsets of the ensemble's reduction / its members' mean KL. Bootstrap over nets for
the shares (2,000, with multiplicities). Runs through the priority queue after E53's last evaluation, before E56's trainings.
**Predictions** (the learned pool makes the nuisance dependence and the variation).
- P227 (identity codes): the code share of uni and of unif is each <= 0.5 x the code share of ctrl.
- P228 (run-to-run variation): c_seed(9) of uni and of unif is each <= c_seed(9) of ctrl - 0.15.
**Also reported (descriptive):** sign shares; c_seed(7) for all arms; the model-based shared and varying parts (E51's
second-order model); the uniform-pool nets' constrained-Bayes cap cost at their c (amendment 63's curve, results/e55_cbkl.json);
a KL budget against unif_noln sq (the exact-statistic floor).
**What the outcomes would mean.**
- P227 and P228 hold: the learned pool is where these nets pick up the arbitrary codes and most of their run-to-run variation;
  with the sign dependence (E53), the main reasons for their distance from Bayes are properties of learned content-based pooling,
  and an architecture that pools a source's tokens by structure rather than by content is predicted to be closer to Bayes.
- P227 fails: the code dependence arises elsewhere (for example in block 2's reading of the summaries).
- P228 fails: uniform-pool nets vary between trainings as much as ctrl nets; their lower KL is a lower shared part.


## Amendment 68 (2026-10-30 20:31Z): E60, is the uniform-pool nets' advantage given source membership or fixed weights? A learned pool restricted to the source's own tokens (architecture; CPU)

Written by the sixth agent before any E60 training. Known beforehand: E1-E58 with their reviews (E53: Update 65; E58: Update 66;
E55: Update 67). In particular E53: nets whose summary tokens pool their own source's four labelled tokens with fixed weights 1/4
(uni) end at KL 0.0044 against ctrl's 0.0127 (ratio 0.35 on 11 paired seeds), sign-flip cost share 0.022 against 0.200, and a
larger departure (rom_large 0.171 against 0.133); the reviewer noted that the uniform pool also hands the net source membership
(which the usual nets must find through the codes) and that the CPU-trained variants were compared with GPU-trained baselines, and
asked for a same-device, source-masked learned-pool control. E22's plumbing: in a trained ctrl net, 98.6% of the summary token's
block-1 attention already falls on its own labelled tokens. E36: the source's own summary route carries most of the negation shift.
E59 (amendment 67) registered, its results not yet seen.
Plumbing (disclosed; step-0 weights only, `e60_plumb.py`): own and ownf have exactly the baseline's parameters at step 0; block 1's
summary rows sum to 1 on their own source's four labelled tokens with non-uniform weights (mean within-source sd 0.06); ownf's query
row has weight 0 on labelled tokens in both blocks, and zeroing the labelled tokens' states after block 1 leaves ownf's output
unchanged (own's changes by 0.8). `e60_summ.py` recovered planted values and decisions on synthetic results with the real comparison
nets.
**Link to M2 (explain the inductive bias; why they deviate).** E53 located most of the standard nets' distance from Bayes in their
pooling architecture, but two things changed at once: the weights within a source became fixed and the source's membership became
given. If a pool that is told the membership but learns its weights is as far from Bayes as ctrl, the cost lies in the learned,
content-dependent weighting of a source's own errors (where E36 found the sign dependence enters); if it is as close as the uniform
pool, the cost lies in finding the membership from arbitrary codes. Either answer names the mechanism an account of the inductive
bias needs.
**Design** (`pfn_own.py`, `train_own.py`, `e60_eval.py`, `e60_summ.py`, `e60_plumb.py`; `pfn_uni.py`, `train_sw.py`, `e48_fmt.py`,
`e48_eval_w1.py`, `e37_eval.py`, `e30_eval.py`, `pfn2.py`, `world.py` unchanged). World 1, ctrl format, amendment 45's fixed 90,000-step
schedule, sparse checkpoints, trained on General's CPU (nice 19, the priority queue, after E56's trainings), every parameter starting
as in the baseline net of the seed, with the same episodes:
- own: in block 1 each summary token may attend only to its own source's four labelled tokens, weights learned (softmax over those
  four); everything else as in pfn2 (seeds 0-11). It differs from E53's uni only in whether those four weights are learned.
- ownf: own, and the query token barred from labelled tokens in both blocks (seeds 0-7); it differs from E53's unif in the same way.
- ctrl_cpu: the baseline on CPU (seeds 0, 1, 4, 7), a device check against E37's GPU-trained ctrl nets of the same seeds.
Measures: `e60_eval.py` (e48_eval_w1.py's: test KL, rom and type, lab, xq, the sign-flip cost). Acquired gap < 0.10; paired seeds
acquired in both; 2,000 bootstrap resamples of seeds, with multiplicities.
**Predictions** (the learned within-source weighting, not membership, makes the cost).
- P229: mean KL(own) / mean KL(uni) >= 2 on paired seeds.
- P230: the pooled sign-flip cost share of own is >= 0.10.
- P231: mean rom_large(own) - mean rom_large(uni) <= -0.02 on paired seeds (the learned weighting partly corrects the cap's
  departure).
**Also reported (descriptive):** ownf against unif and against own; own against ctrl and ctrl_cpu; ctrl_cpu against E37's ctrl
(KL ratio and rom per seed); types, lab, xq; the spread of KL and rom across seeds.
**What the outcomes would mean.**
- P229-P231 hold: with membership given, learned within-source weights bring back most of ctrl's KL and sign dependence and part of
  its correction of the departure; the cost is in the learned weighting of a source's own errors, not in finding its tokens.
- P229 and P230 fail (own close to uni): the uniform pool's advantage is the membership it hands over; the cost of the standard
  nets lies in finding a source's tokens through arbitrary codes.
- Mixed (for example KL close to uni but a sign share above 0.10): the two costs separate.
- The device check: if ctrl_cpu differs from E37's ctrl by much more than float rounding would suggest (KL ratio outside about 0.8-1.25
  or rom differences above about 0.02 on most seeds), E53's comparisons with GPU-trained baselines need to be redone on one device.


## Amendment 69 (2026-10-30 21:07Z): E61, do the uniform pool's advantage and the constrained-Bayes prediction hold at width 128 and depth 3? (architecture across sizes; GPU, about $1.2)

Written by the sixth agent before any E61 training. Known beforehand: E1-E59 with their reviews (Updates 63-68), in particular
E53 (width 32: uniform-pool nets at KL 0.0044-0.0046 against ctrl's 0.0127, ratio 0.35-0.36; departure 0.17-0.18), E55 (their
departure within 0.01 of constrained Bayes at their learned c of about 1.4; for learned pools c is larger at width 128), E58 (with
a uniform pool, setting c by design moves the departure as constrained Bayes predicts), E59 (their code dependence, sign dependence
and run-to-run variation are small), E38 and E52 (ctrl's KL falls to about half at width 128; depth 3 lowers mainly the shared
part). E60 registered and queued (CPU), no result.
Plumbing (disclosed; not results): pfn_uni's unif variant at width 128 depth 2 and at width 32 depth 3 has exactly E38's ctrl nets'
step-0 parameters (seed 0) and trains (one forward and backward pass); `e61_summ.py` recovered planted decisions on synthetic
results (with E38's nets standing in for the unif nets).
**Link to M2 (across sizes; predict the departure from the architecture).** M2 asks for an account that holds across sizes and
predicts how the departure scales. E53-E59 established, at width 32 depth 2, that a structural uniform pool brings the nets about
three times closer to Bayes and that its departure follows constrained Bayes at the learned cap scale. If both hold at four times
the width and at one more layer, the uniform-pool account is not a small-net artefact; how c, and with it the predicted departure,
changes with size gives the scaling that M2 asks for.
**Design** (`gpu/jobs_e61.txt`, `e61_summ.py`; `train_uni.py`, `pfn_uni.py`, `e53_eval.py`, `lngeom.py`, `train_sw.py`, `e48_fmt.py`,
`pfn2.py`, `world.py` unchanged; GPU scripts `gpu/gpu_jobs_x.sh`, `gpu/launch_x.sh`, `gpu/retry_x.sh`). The unif variant (uniform
within-source pool, query barred from labelled tokens in every block), ctrl format, amendment 45's fixed 90,000-step schedule,
sparse checkpoints, seeds 0-9, at width 128 depth 2 and at width 32 depth 3; every parameter starts as in E38's ctrl net of the same
seed and size, with the same episodes and on the same kind of device (L4). 20 trainings on one on-demand L4 (8 at once), estimated
$1.2; lane GPU total about $34.8 of $40. Evaluation on CPU with `e53_eval.py`; c of each net by lngeom.py as e55_run.py; constrained
Bayes from amendment 63's table. Acquired gap < 0.10; paired seeds acquired in both; 2,000 bootstrap resamples of seeds.
**Predictions.**
- P232: mean KL(unif) / mean KL(ctrl) <= 0.6 on paired seeds, at width 128 and at depth 3.
- P233: at width 128 and at depth 3, the unif nets' mean rom_large and rom_middle are each within 0.03 of the mean constrained-Bayes
  value at their c.
**Also reported (descriptive):** c by size (median, range, A's spread, rho); sign shares; KL and rom against E53's width-32 unif nets
(different device); stalls.
**What the outcomes would mean.**
- Both hold: the uniform pool's advantage and its constrained-Bayes departure persist over a fourfold width and an extra layer;
  the departure's size at each size follows from the learned cap scale.
- P232 fails: the learned pool's extra cost shrinks with size (wider or deeper standard nets catch up), so the advantage is a
  small-net effect.
- P233 fails: at larger sizes the uniform-pool nets' departure is no longer the cap's (for example later layers re-weight the pooled
  evidence), so the constrained-Bayes account holds only for the 2-layer width-32 nets.


## Amendment 70 (2026-10-31 03:53Z): E62, does given source membership also remove the code dependence and the run-to-run variation? E59's measures for the learned within-source pools and the fixed-code nets (CPU, existing nets)

Written by the seventh agent before any E62 quantity is computed on a trained net. Known beforehand: E1-E61 with their reviews
(Updates 1-71). In particular:
- E59 (amendment 67): code share (the KL reduction of geometric averaging over 8 redraws of the three source codes, as a share of
  the family KL) 0.012 for uni and unif against ctrl's 0.276; c_seed(9) 0.20 and 0.18 against ctrl's 0.55 (c_seed(7) 0.20, 0.19
  against 0.50); E51's model-based split at K = 7: ctrl shared 0.0051 + varying 0.0075, unif 0.0040 + 0.0011.
- E60 (amendment 68): told the membership, learned within-source pools reach the uniform pool's KL (own 0.0042, sd 0.0004; ownf
  0.0042) with sign shares 0.058 and 0.047, all A-type, rom_large 0.018 and 0.048. The reviewer noted that E59's code and run-to-run
  measures were not repeated for these nets.
- E30 (amendment 38): fix nets (the same three source codes and five question codes in every episode; routing to a source's tokens
  is still by content, through those codes): all 12 acquired; this agent printed their per-seed test KL (0.0038-0.0086, mean
  0.0064), rom_large (0.05-0.25; 8 of 12 B-type by the middle-range sign) and sign-family costs before this amendment.
Plumbing (disclosed; not results): `e62_plumb.py` built a scratch tree (/tmp/e62_plumb) with the step-0 (untrained) weights of the
own and ownf nets and, for fix (whose runs saved no checkpoints), E37's ctrl step-0 weights of the same seeds (the same
initialisation, E30's pairing), with placeholder evaluation files; `e62_run.py` (60 bases) and `e62_summ.py` ran end to end there.
No trained net's E62 quantity was computed. The summary's plumbing run printed, from existing registered files: the sign shares of
own (0.058), ownf (0.047) and fix (0.090 [0.055, 0.137], from E30's registered evaluations) and, from E59's per-net values, ctrl's
code share by type (A 0.216 over 4 nets, B 0.290 over 7), both descriptive quantities of this amendment.
**Link to M2 (why they deviate; the inductive bias).** E60 showed that, told the membership, a learned pool reaches the uniform
pool's KL; the uniform pool's other advantages (E59: almost no dependence on the arbitrary source codes, much smaller variation
between trainings) were not measured for it. If given membership also removes those, then the excess KL, the code dependence and the
run-to-run variation are costs of one thing, finding a source's tokens by content through random codes, which a routing account must
then explain quantitatively (amendment 71 is planned to test its mechanism at evaluation, and retraining tests are planned after
it). If the own nets keep ctrl's code dependence or variation, those costs come from elsewhere (the learned within-source weighting,
the unrestricted rows, block 2). The fix nets separate random-code geometry from content routing as such: their routing is by
content, but their codes never change.
**Design** (`e62_run.py`, `e62_summ.py`, `e62_plumb.py`; E59's estimator and code unchanged: `e26_run.py`, `e29_data.py`, `e30_eval.py`,
`e48_fmt.py`, `pfn_own.py`, `pfn_uni.py`, `pfn2.py`).
- Arms: E60's own (12 nets) and ownf (8), E30's fix (12); references from E59's outputs (`results/e59/`): ctrl (E37, 11 acquired),
  uni and unif (12 each). Acquired: registered gap < 0.10.
- Per net, exactly as E59: base KL on E27's first 3,000 bases (±6 grid); for own and ownf, the identity-code family (E29's SID family,
  6,000 bases × 9: the base and 8 redraws of the three source codes): its mean KL and the KL reduction of geometric averaging; not for
  fix (its format replaces every code by the fixed ones, so the family is constant). Sign shares from the registered evaluations.
- Seed ensembles: K = 7 for every arm, K = 9 for own and fix; 5 random K-subsets (numpy seed 62); c_seed(K) = mean over subsets of the
  ensemble's KL reduction / its members' mean KL; E51's shared and varying parts. Shares with 2,000 bootstrap resamples of nets
  (multiplicities).
- Runs: one job per arm in the priority queue (`dynq.sh`, nice 19, ionice -c3), then `e62_summ.py`.
**Predictions** (finding membership by content makes the code dependence and the run-to-run variation; random code geometry makes
the variation).
- P234: code share(own) <= 0.5 x code share(ctrl) and code share(ownf) <= 0.5 x code share(ctrl) (E59's bar; ctrl 0.276).
- P235: c_seed(9)(own) <= c_seed(9)(ctrl) - 0.15 and c_seed(7)(ownf) <= c_seed(7)(ctrl) - 0.15 (ctrl 0.55 and 0.50).
- P236: c_seed(9)(fix) <= c_seed(9)(ctrl) - 0.15.
**Also reported (descriptive):** base KL, sign shares, the shared and varying parts at K = 7 and 9, types, and ctrl's code share by
type. The fix nets were trained on GPU (E30) and the own nets on CPU (E60's device check found the ctrl endpoints equal within about
1% on four seeds); E59's ctrl nets on GPU.
**What the outcomes would mean.**
- P234 and P235 hold: given membership removes the code dependence and the run-to-run variation as well as the excess KL; with E59
  and E60, every measured advantage of the uniform architecture comes from handing over membership, and finding membership by content
  is where the standard nets acquire them.
- P234 fails: the own nets depend on the codes although membership is given, so the code dependence is made elsewhere.
- P235 fails: the variation between trainings survives given membership; with E59 (uniform pools removed it), it then comes from the
  learned within-source weighting.
- P236 holds: with fixed codes the variation is much smaller, so random code geometry (overlaps, norms and directions that change in
  every episode) makes most of it. P236 fails: content routing makes it even when the codes never change (for example through two
  solution types, or LayerNorm's coupling of routing and evidence features).


## Amendment 71 (2026-10-31 04:06Z): E63, how do the standard nets' block-1 summary heads route and pool? Head allocation, the error dependence of the pooling logits, leakage, the code-set normaliser and code idealisation at evaluation (CPU, existing nets)

Written by the seventh agent before any E63 quantity is computed on a registered net. Known beforehand: E1-E61 with their reviews and
E62's results (P234-P236 hold; Update 72 in review). In particular: E22 (in trained ctrl-type nets 98.6% of the summary token's block-1
attention falls on its own labelled tokens; the within-source share against |error| has Spearman median -0.09), E27 (B bases'
labelled tokens attend across sources on the same question in block 1, xq 5-8% against 0.2-0.4%), E32 (switching that labelled-row
attention off changes the KL by about 1%), E44 (block 1's attention weights at initialisation transfer the type most often), E47 (xq
and the sign share separate the types at 10,000 steps, AUC 0.97), E30 (fix: 4 A / 8 B), E60 (own: all 12 A).
**Exploration (disclosed; not results).** `e63x_explore.py`, `e63x_explore2.py` and `e63_run.py`'s plumbing cohort, on E43's newinit nets
s0, s1, s2, s4 and s11 (types A, B, B, B, A), which every registered test below excludes:
- In the three B nets one of the two summary heads puts about 0.33 on the summary token itself and 0.66 on the other two sources'
  summary tokens (C = 0.98-0.99) and almost nothing on labelled tokens; the other head pools the source's own labelled tokens (P =
  0.81-0.82). In the two A nets both heads pool (P = 0.99; C = 0.001).
- Pooling heads: norm-scaled own logits L_own +16 to +23 in all 7 pooling heads (other sources' labelled keys +6 to +9); the implied
  logit change of an own key at |r| = 3 against r = 0, -3.3 to -3.9; the odd coefficient K_odd has RMS 1.6-2.1, and in both A nets
  the two pooling heads' mean K_odd have opposite signs (-2.11 and +1.55; +1.75 and -1.90).
- Leakage of the pooling heads: mean 0.010-0.025, 99% quantile 0.19-0.31, Spearman with the source code's largest cosine 0.46-0.51.
- Per-source A CV 0.13-0.16 with the real codes, 0.0001-0.004 with both codes zeroed; with equal-norm codes 0.10-0.12.
- KL ratio orthn / base 0.87-0.95 (eqn 0.92-1.01, orth 0.94-0.98) on 1,000 and on 300 bases.
- Two evaluation-time surgeries, not used below: restricting the summary rows to their own source's tokens lowered the A net's KL by
  9% and multiplied the B nets' by 16-49 (their consensus head is then forced onto other tokens); replacing block 1's LayerNorm
  denominator by its expectation over the codes multiplied the KL by 3.5-10 (co-adaptation).
- Also seen: the per-net code costs of E62's own and ownf nets in its job logs (amendment 70's outputs).
**The idea (pre-LN attention algebra).** In block 1, the logit of key j for a summary query is, up to a constant common to all keys,
logit_j = (alpha_j + beta r_j) / n_j, with alpha_j = sqrt(d / dh) q.W_K(gamma * a_j) (the token at r_j = 0), beta = sqrt(d / dh)
q.W_K(gamma * u~) and n_j = |a_j + r_j u~| the LayerNorm norm (lngeom.py's decomposition). Its dependence on the size of the error
therefore enters only through the norm (an even term proportional to alpha_j, which shrinks as |r_j| grows) and through the signed
feature (an odd term proportional to beta). Routing by content needs the own keys' logits far above the other sources'; if alpha_own
is then positive, a large error loses weight (L_own = alpha_own / A is the norm-scaled logit). A head can recover an even, roughly
quadratic dependence on the errors only by pairing the odd logit term with the odd part of the values (the weighted mean of r_j / n_j
under weights exp(beta r_j / n_j) is about beta sum r_j^2 / n_j^2), which is sign-dependent within one head and sign-symmetric
across two heads with opposite beta. Under this reading the A type pools with both summary heads, with opposite odd coefficients,
and the B type spends one head on a consensus over the three sources' query readings and pools with one sign-dependent head; with
membership given (own) the norm-scaled logit no longer has to carry the routing.
**Link to M2 (explain the inductive bias; predict the departure from the architecture).** E60 and E62 place the standard nets' main
costs in finding a source's tokens by content. This episode measures, on fresh nets, what the summary heads of block 1 do with the
routing: which heads route and pool, how the pooling logits depend on the errors (an exact consequence of pre-LN attention), how much
the routing leaks and how the codes set the LayerNorm normaliser, and what an ideal code geometry gains at evaluation. If the A/B types
are two allocations of these heads and the pooling logits' error dependence follows the routing requirement, the program's
solution types, sign dependence and part of the departure follow from one architectural constraint, which then predicts what changes
them (more heads, codes outside the normalisation, sign-free inputs) for testing by retraining.
**Design** (`e63_run.py`, `e63_summ.py`, `e63_plumb.py`; `e26_run.py`, `e30_eval.py`, `e48_fmt.py`, `pfn_own.py`, `pfn_uni.py`, `pfn2.py`,
`world.py` unchanged). E27's first 2,000 bases (variant 0), each net in its own format, best-validation model.pt.
- Cohorts: ctrl = E37's 11 acquired ctrl nets; fix = E30's 12 fix nets (fixed codes); own = E60's 12 own nets; ext = the acquired final
  nets of E47's E43, E44 and E46 lineages except the five explored newinit nets (clusters = E47's parent seeds); traj = E37's 11
  acquired ctrl runs at steps 0, 2,000, 3,000, 5,000, 10,000, 20,000, 40,000, 60,000, 80,000, 90,000 (M1 only). Types: the registered
  final middle-range rom sign.
- M1 (baseline architectures): per summary head h, the block-1 attention mass on the source's own labelled tokens (P_h), on itself, on
  the other two summary tokens (osum_h) and on other sources' labelled tokens; C_h = self + osum_h; consensus head: max_h C_h >= 0.5;
  pooling head: P_h >= 0.5.
- M2 (ctrl): the pooling heads' cross-source mass per (episode, source), its mean and quantiles, and its Spearman correlation with the
  source code's largest cosine with another source's code.
- M3 (ctrl, fix, own): per-source A CV (the mean of A_j over the source's four labelled tokens), with the real codes and with both
  codes zeroed.
- M4 (ctrl, own): KL with the source codes changed at evaluation (Bayes-invariant): eqn (norm 1), orth (Loewdin-orthogonalised, norms
  kept), orthn (orthonormal), against the base codes on the same bases.
- H3 (ctrl, fix, own, ext; pooling heads, for own both heads): L_own, L_oth, the implied logit change at |r| = 3 (dlogit; A / n
  averaged over both signs, with each token's A, U and rho), and K_odd = sqrt(d / dh) q.W_K(gamma * u~) / A_j (RMS and mean over own
  keys).
- Runs: one job per cohort in the priority queue (nice 19), then `e63_summ.py`.
**Predictions.**
- P237 (head allocation and type): in E37's ctrl nets and E30's fix nets, a consensus head is present in every B-type net and in no
  A-type net, with at most one exception among the 23 nets.
- P238 (the same on other nets): over ext, AUC(C_max; B > A) >= 0.90 (interval by 2,000 cluster-bootstrap resamples).
- P239 (the sign-symmetric pair): among the A-type nets of ctrl and fix with two pooling heads, the two heads' mean K_odd have opposite
  signs, with at most one exception.
- P240 (the routing's even down-weighting): (a) in ctrl, every pooling head has mean L_own > 0 and mean dlogit(|r| = 3) < -1, with at
  most one exception over the cohort's pooling heads; (b) the median of own's L_own over its 24 heads is < 0.5 x the median over
  ctrl's pooling heads.
- P241 (code geometry at evaluation): in ctrl, KL(orthn) < KL(base) in >= 9 of 11 nets and the mean of 1 - KL(orthn) / KL(base) is
  >= 0.03; in own, the mean of |1 - KL(orthn) / KL(base)| is <= 0.02.
**Also reported (descriptive):** C_max over training (median by final type, AUC by step, against E47's markers); n_pool; the per-type
leakage, A CV, K_odd RMS, dlogit and KL ratios (eqn, orth, orthn); fix and ext's H3 quantities; own's A CV and K_odd pairing.
**What the outcomes would mean.**
- P237 and P238 hold: the A/B solution types are two allocations of block 1's summary heads, two pooling heads (A) or one pooling head
  and one head that averages over the sources' query tokens (B); with E44, the type is the role each head takes from its initial
  weights. P237 holds and P238 fails: a property of these lineages only. P237 fails: the exploration's pattern does not generalise.
- P239 holds: A's two pooling heads carry opposite odd coefficients, the sign-symmetric pair of the idea above; with E27's sign costs
  (A 7%, B 25%), B's single pooling head would be the source of its larger sign dependence (a hypothesis for a causal test). Fails:
  pairing is not what separates the A nets' pooling from the B nets'.
- P240 holds: the standard nets' pooling heads down-weight large errors through LayerNorm because their routing logit is positive,
  and with membership given the nets do not do this; this is a second route to the big-miss departure, made by routing. (a) fails:
  the routing does not force that down-weighting; (b) fails: own nets keep similar norm-scaled logits, so it is not a routing
  requirement.
- P241 holds: an ideal code geometry helps the trained standard nets by a few percent and leaves own unchanged, so random code
  geometry costs them something, though much less than the code-redraw averaging gain (28%); fails: the trained nets do not profit
  from cleaner codes at evaluation (the code dependence is not about the raw codes' overlaps and norms).


## Amendment 72 (2026-10-31 04:21Z): E64, content routing without its two side effects: no summaries head in block 1 (nocons), the codes outside block 1's LayerNorm (codeln), and both (architecture; trained on CPU)

Written by the seventh agent before any E64 training (the 1,000-step smoke runs below excepted). Known beforehand: E1-E63 with their
reviews (E62: Update 72; E63: Update 73). In particular:
- E63: an A net pools its source's own errors with both block-1 summary heads, a B net with fewer (post hoc, all 222 nets); with
  random codes B's other head puts its mass on the three summaries (129 of 132 ext B nets; registered P237 failed on the fixed-code
  nets, where B's other head often attends to other sources' labelled tokens); every run starts with such a head and A runs convert it
  at 5,000-20,000 steps. In every standard pooling head the norm-scaled own logit is large and positive (L_own 14-22; the norm term
  lowers an own key's score at |r| = 3 by about 3.5 nats); in own nets it is slightly negative (median -0.82), in fix nets 2.9-5.3.
  The per-source LayerNorm norm varies by 13-15% through the codes in the standard nets. Orthonormal codes at evaluation lower the
  standard nets' KL by 14%.
- E60, E62: told the membership (own), learned pools reach KL 0.0042, rom_large 0.018, code share 0.016, c_seed(9) 0.20, all A-type;
  E37's ctrl (GPU) 0.0127, 0.133, 0.276, 0.55, 4 A / 7 B and one stall (seed 3).
Plumbing (disclosed; not results; `e64_plumb.py`, smoke runs): nocons and codeln nets built with the seed have exactly E37 ctrl's
step-0 parameters (seeds 0, 1); nocons puts zero block-1 attention from a summary token on the other summaries and leaves every other
block-1 row equal to the baseline's; codeln's normaliser does not change when the codes are rescaled, and with zero codes it computes
the baseline's function; forward and backward passes run. 1,000-step CPU smoke runs of seed 0 (`--const 1000 --cooldown 0`; on CPU
these are exactly the first 1,000 steps, all in warmup, of the registered seed-0 runs): validation gaps 0.212 (nocons) and
0.238 (codeln), not yet acquired; `e64_heads.py` ran on them (no pooling head yet at 1,000 steps) and reproduced E63's values on an
explored net (L_own 20.67 and 18.13 against 20.65 and 18.11); `e64_summ.py` ran end to end on synthetic inputs (copies of E37's
ctrl evaluations).
**Link to M2 (predict the departure and the cost from the architecture; test the account causally).** E63 read the standard nets'
solution types, B's sign dependence and part of their departure as consequences of two properties of block 1: a head can spend itself
on the summaries (the B allocation), and, in pre-LN attention, the routing score that finds a source's tokens by their codes also
scales with 1 / n_j, so it lowers large errors' scores. Each change below removes one of these while keeping content-based routing
(the nets must still find a source's tokens by matching random codes, unlike E60's own). If the account is right, removing both should
bring content-routing nets close to the nets told the membership; whatever remains is the cost of routing as such (leakage, code
geometry). An architecture-level statement of this kind (routing codes kept out of the normalisation, enough heads for pooling)
would predict the standard architecture's excess from its design.
**Design** (`pfn_e64.py`, `train_e64.py`, `e64_eval.py`, `e64_heads.py`, `e64_codes.py`, `e64_summ.py`, `e64_plumb.py`; `train_sw.py`,
`e48_fmt.py`, `e48_eval_w1.py`, `e37_eval.py`, `e30_eval.py`, `e62_run.py`'s estimator, `pfn_own.py`, `pfn_uni.py`, `pfn2.py`, `world.py`
unchanged).
- Arms (world 1, ctrl format, amendment 45's fixed 90,000-step schedule, train_sw.py's recipe, sparse checkpoints, seeds 0-11; every
  parameter starts as in E37's ctrl net of the seed, with its episodes; General's CPU through the priority queue, nice 19, at most 3
  trainings at once so that evaluations keep a slot):
  - nocons: block 1's summary tokens may not attend to the other sources' summary tokens (everything else as pfn2);
  - codeln: block 1's pre-attention LayerNorm normalises only the content part of each token (the o, y, missing and query-flag columns
    and the bias); the code part W[:, :32] X[:, :32] is added after it; the residual stream and block 2 as pfn2;
  - both: codeln and nocons together.
  36 trainings.
- Measures: `e64_eval.py` (e48_eval_w1.py's: test KL, rom large and middle, type, lab, xq, the sign-flip cost); `e64_heads.py` (E63's
  head measures with the variant's own block 1: P_h, C_h, pooling heads, L_own and dlogit from the part the LayerNorm normalises,
  K_odd, A's CV; for codeln and both the code logits); `e64_codes.py` (E59's code share and seed ensembles, K = 7 and 9). Acquired:
  gap < 0.10. Comparisons: E37's ctrl (GPU) and E60's own (CPU) nets of the same seeds; E63's ctrl pooling heads; E59's ctrl code share.
  Paired statistics over seeds acquired in both arms, 2,000 bootstrap resamples of seeds (multiplicities).
**Predictions.**
- P242 (the summaries head makes the B allocation): >= 80% of the acquired nocons nets have two pooling heads and are A-type.
- P243 (the codes in the normaliser make the routing logit lower large errors' scores): (a) the median over codeln's pooling heads of
  L_own (from the normalised content part) is <= 0.5 x the median over E63's ctrl pooling heads (16.8); (b) the paired mean rom_large
  of codeln is <= 0.7 x ctrl's.
- P244 (without both side effects, content routing costs little): the paired mean KL of both is <= 1.3 x own's, and its mean
  rom_large is <= 0.06.
- P245 (each side effect costs KL): the paired mean KL ratios nocons / ctrl and codeln / ctrl are each <= 0.75.
- P246 (the code dependence goes through the normaliser): the code shares of codeln and of both are each <= 0.5 x ctrl's (0.276).
**Also reported (descriptive):** stalls; types; pooling-head counts, consensus heads, K_odd pairing, dlogit, code logits; sign shares;
lab and xq; c_seed(9); nocons / codeln / both against own (KL, rom); per-seed tables.
**What the outcomes would mean.**
- P242 holds: without the summaries option the nets settle in the A allocation; the B type is a choice between pooling and looking at
  the summaries. Fails: B re-forms through other cross-source heads (as in fix), so the allocation is not about the summaries as such.
- P243 holds: taking the codes out of the normalisation frees the routing from lowering large errors' scores and shrinks the
  departure; with E63 this makes the 1 / n_j scaling of the routing score a cause of part of the standard nets' departure. (a) holds and
  (b) fails: the routing logit's sign changes but the departure does not follow, so that term is not what holds the departure up.
- P244 holds: content routing as such is cheap; the standard architecture's excess is made by the two side effects. Fails: routing by
  content costs KL beyond them (leakage, code geometry, learning).
- P245 and P246 measure each side effect's share of the KL and of the code dependence.


## Amendment 73 (2026-10-31 04:26Z): E65, do the head allocation and the routing logits hold across sizes and in the sign-free format? E63's measures on E38's and E37's nets (CPU, existing nets)

Written by the seventh agent before any E65 quantity is computed on a registered net; E64 (amendment 72) is training and none of its
outputs exists. Known beforehand: E1-E63 with their reviews; in particular E63 (Update 73): in the width-32 ctrl cohorts an A net has
two block-1 summary pooling heads and a B net fewer (post hoc, 222 of 222), A's two pooling heads carry opposite signed coefficients,
every standard pooling head has a large positive norm-scaled own logit (L_own 14-22; the norm term about -3.5 nats at |r| = 3, median
-3.31 over E63's ctrl pooling heads); E38 (Update 51) and E55: at width 64, width 128 and depth 3 the ctrl and mag nets' types and
departures (rom_large ctrl 0.122, 0.064, 0.089; mag 0.079, -0.002, 0.040; E37's mag nets 11 of 12 A-type). No head or logit quantity
of these nets has been computed.
Plumbing (disclosed; not results): `e65_run.py` ran on E63's explored nets (E43 newinit s0, s1, s2, s4: the values reproduce E63's
exploration) and on E37 mag s0's step-0 (untrained) weights in a scratch tree (/tmp/e65_plumb), where the |r| = +3 ratio and the
logit decomposition run; `e65_summ.py` ran end to end on copies of the explored nets' output.
**Link to M2 (across sizes and formats; predict from the architecture).** E63's reading of the solution types (head allocation) and of
part of the departure (the routing logit's 1 / n_j scaling) was found at width 32 and depth 2 in the signed format. M2 needs it across
sizes, where the departure shrinks with width (0.133 to 0.064), and the account makes a format prediction: in the sign-free format the
o column carries |r|, an even feature, so a single head can raise large errors' scores through the linear term without depending on
the signs, which fits that format's almost uniformly A-type nets. If the allocation rule and the routing logits hold across sizes,
and the width-128 nets' norm term is weaker where their departure is smaller, the account covers the size dependence; if not, it is a
property of the small nets.
**Design** (`e65_run.py`, `e65_summ.py`; `e63_run.py`'s functions, `e30_eval.py`, `e48_fmt.py`, `pfn_own.py`, `pfn_uni.py`, `pfn2.py`
unchanged). Cohorts: E38's ctrl nets at width 64, width 128 and depth 3, E37's mag nets (width 32; the mag60 arm excluded) and E38's mag
nets at the three sizes; acquired = registered gap < 0.10; types from the registered evaluations. Per net on E27's first 2,000 bases,
E63's M1 and H3 for the summary tokens' block-1 heads (all these nets have two heads); in the mag format the coefficient of |r| (K_lin)
and the norm term at |r| = +3 only. One job per cohort in the analysis queue (one slot while E64 trains), then `e65_summ.py`.
**Predictions.**
- P247 (allocation across sizes): in each of the three E38 ctrl cohorts, A-type if and only if two pooling heads, with at most one
  exception per cohort.
- P248 (routing logits across sizes): every pooling head of the three ctrl cohorts has mean L_own > 0 and mean dlogit(|r| = 3) < -1,
  with at most 10% exceptions over their pooling heads.
- P249 (the sign-free format): in each of the four mag cohorts, A-type if and only if two pooling heads (at most one exception per
  cohort), and the mean K_lin is > 0 (large |r| raises the score through the linear term) in >= 80% of their pooling heads.
- P250 (width and the norm term): the median dlogit over the width-128 ctrl pooling heads is less negative than E63's width-32 median
  (-3.31).
**Also reported (descriptive):** consensus heads, pooling-head counts, L_own, dlogit, K RMS and sign pairing by cohort and type; the
mag nets' K pairing.
**What the outcomes would mean.**
- P247 and P248 hold: the head-allocation reading of the types and the routing logits' norm term are properties of this
  architecture and recipe across a fourfold width and an extra layer, not of the width-32 nets only. Fail: they are size-specific.
- P249 holds: with an even error feature the A nets still pool with two heads, but raise large errors' scores through the linear term;
  the pairing of signs in the signed format is then a way of getting an even statistic that the sign-free format supplies directly.
  Fails on K_lin: the sign-free nets do not use the linear term that way.
- P250 holds: wider nets' norm term is weaker, in the direction of their smaller departure; fails: the departure's fall with width
  does not go through this term.


## Amendment 72, addendum 1 (2026-10-31 04:38Z): E64b, the corrected arm: split normalisation of block 1's input (split) and split with nocons (splitnc), on one L4 (coordinator-approved, cap $2)

Written by the seventh agent before any E64b training (the smoke run below excepted). E65 is written up (Update 74); E64's registered
arms are training on CPU.
**Why the codeln arm is flawed (stated before any of its results is analysed; its registered arms and results are kept).** codeln adds
the code part of the embedding after block 1's LayerNorm without normalising it. At initialisation that part has norm 0.75 in block
1's input (seeds 0 and 1), against 3.5-3.6 for the code part of the baseline's LayerNorm output, and its ratio to the content part is
0.13 against 0.81-0.84: the routing signal starts about five times weaker in absolute size and six times weaker relative to the
content. Seen before this addendum: E64's seed-0 codeln run sat on a plateau (validation gap 0.14-0.15) from 2,000 to 24,000 steps,
and the seed-0 both run until about 22,000 steps (0.113 at 24,000), where the baseline escapes at about 4,000-6,000 and nocons by
3,000 (0.085). So codeln tests "codes outside the normalisation" together with "a much weaker code channel", and its stalls or delays
would not bear on P243-P246's mechanism. Those predictions stay registered and will be reported as they come out, with this caveat.
**The corrected arms** (`pfn_e64b.py`, `train_e64b.py`, `e64b_eval.py`, `e64b_heads.py`, `e64b_codes.py`, `e64b_summ.py`, `e64b_plumb.py`;
`pfn_e64.py`'s attention helper and nocons mask unchanged). Block 1's pre-attention input is h = gamma * (N(x_c) + N(x_k)) / sqrt(2) +
beta, with x_c the content part (o, y, missing and query-flag columns and the bias), x_k the code part, N a LayerNorm without affine
(centring and division by the root mean square, with LayerNorm's eps), and gamma, beta block 1's LayerNorm parameters: the error
feature's normaliser contains no code and the codes' scale contains no error, and both parts start at norm 4.0 (the baseline's code and
content parts 3.5-3.6 and 4.3-4.4). The code part is also normalised to a fixed norm per token, which removes the random code length.
- split: that block 1; everything else as pfn2. splitnc: split and nocons.
- No parameter is added; with the seed every parameter starts as in E37's ctrl net, with its episodes. World 1, ctrl format, amendment
  45's fixed 90,000-step schedule, sparse checkpoints, seeds 0-11: 24 trainings on one on-demand L4 (8 at once; `gpu/launch_x.sh`,
  `gpu/retry_x.sh`), estimated $1.3, cap $2 (the coordinator's approval, received at 04:38Z); the VM is deleted by the runner, checked in the instance list and
  logged in GPU-LOG.md. E37's ctrl nets were trained on L4 too; own on CPU.
- Measures and comparisons as amendment 72 (e64b_eval.py, e64b_heads.py, e64b_codes.py), on CPU after the runs.
Plumbing (disclosed; not results; `e64b_plumb.py`): split and splitnc nets have exactly E37 ctrl's step-0 parameters (seeds 0, 1); the
content part of h is unchanged when every code is rescaled (4.8e-7), and its code part is unchanged when the values change (3.6e-7);
both parts have norm 4.0 at initialisation; splitnc puts zero block-1 attention from a summary token on the other summaries; forward
and backward passes run. A 6,000-step CPU smoke run of split seed 0 (`--const 6000 --cooldown 0`; the registered run's first 6,000
steps): validation gaps 0.243, 0.105, 0.087, 0.084, 0.052, 0.064 at 1,000-6,000 steps, so split escapes the plateau about as early
as the baseline.
**Predictions** (as amendment 72's P243-P246 for the corrected arms; from E63-E65's head-level account: with the codes out of the
content's normaliser, the norm term need not lower large errors' scores, so a single pooling head can raise them for both signs
through a negative norm-scaled logit, as nets told the membership do).
- P251 (split, the norm term and the departure): (a) the median over split's pooling heads of L_own (content part) is < 0; (b) the
  paired mean rom_large of split is <= 0.7 x ctrl's.
- P252 (splitnc, routing without the side effects): its paired mean KL is <= 1.3 x own's, and its mean rom_large is <= 0.06.
- P253: the paired mean KL ratio split / ctrl is <= 0.75.
- P254: the code shares of split and of splitnc are each <= 0.5 x ctrl's (0.276).
- P255 (types): >= 75% of the acquired split nets are A-type.
**What the outcomes would mean.** As amendment 72 for codeln and both. In addition: P251 (a) and P255 hold: with the codes out of the
error feature's normaliser the nets choose pooling that raises large errors' scores through the norm term and end A-type, whatever
their head allocation, which makes the routing logit's 1 / n_j scaling the reason the signed-format nets need an opposite-sign pair.


## Amendment 74 (2026-10-31 04:45Z): E66, what does the signed term of the pooling logits do? Removing it from one or both pooling heads at evaluation (CPU, existing nets)

Written by the seventh agent before any E66 quantity is computed on a registered net. Known beforehand: E1-E65 with their reviews
(E65: Update 74); E64 and E64b registered and training, no output.
**The question.** E63-E65 found that a standard net's block-1 pooling-head logits have the form (alpha_j + beta r_j) / n_j + const,
that A nets pool with two heads whose signed coefficients have opposite signs, and that B nets pool with one. Whether the signed term
carries much of the pooling's reliability computation (a weighted mean of r_j / n_j under weights depending on beta r_j carries an
even, roughly quadratic statistic of the errors), and whether the pair's opposite signs cancel the sign dependence, was not tested.
**Exploration (disclosed; not results; on E63's explored nets E43 newinit s0 and s11 (A) and s1 and s4 (B), which every test below
excludes; 1,000 bases).** Removing one pooling head's signed term multiplied the KL by 8-13 in the A nets and by about 15 in the B
nets; the sign share rose from 0.07 to 0.43-0.48 in the A nets and from 0.30-0.32 to 0.40-0.41 in the B nets; removing both of an A
net's terms multiplied the KL by about 50 and gave a sign share of 0.12-0.14, below the single removals. Plumbing (`e66_plumb.py`):
the surgery with no head reproduces each net's forward exactly (also for an own net at step 0), changes only the chosen head's
summary-row logits of labelled keys, and removes exactly E63's term (relative difference 2e-5, LayerNorm's eps); `e66_summ.py` ran on
the exploration numbers.
**Link to M2 (how they compute; why they deviate).** If removing the signed term wrecks the pooling, the learned pools compute the
source's reliability largely through it, an estimator that is sign-dependent within each head by construction; with A's pair, the
sign dependence of the single remaining head would show what the pair cancels. That would explain, from the attention algebra, why
standard nets depend on the error signs that Bayes ignores, and why the A type depends on them less.
**Design** (`e66_run.py`, `e66_summ.py`, `e66_plumb.py`; `e26_run.py`, `e27_data.py`, `e30_eval.py`, `e37_eval.py`, `e48_fmt.py`,
`pfn_own.py`, `pfn_uni.py`, `pfn2.py` unchanged). Nets: E37's ctrl and E30's fix nets with at least one pooling head (E63's pooling
heads; ctrl: 4 A with two, 6 B with one; fix: 4 A with two, 3 B with one) and E60's 12 own nets (both heads pool). Surgery: for the
summary rows of a chosen head, logit_j -> logit_j - q.W_K(gamma u~) r_j / (sigma_j sqrt(dh)) for the labelled keys j (the exact
signed term, sigma_j the LayerNorm standard deviation with eps). Conditions: base; each pooling head alone; all pooling heads. Per
condition on E27's first 3,000 bases: KL; the sign-flip family's KL reduction (sign_pc) and mean KL (sign_kl); sign share =
sign_pc / sign_kl; rom on the E12 pairs for base and all-zeroed. One job per cohort in the analysis queue (nice 19).
**Predictions.**
- P256 (the signed term carries the pooling): in ctrl and fix, removing one pooling head's signed term multiplies the KL by >= 3 in
  >= 80% of the (net, head) cases.
- P257 (the pair cancels sign dependence): in the A nets of ctrl and fix, removing one head's signed term at least doubles the sign
  share in >= 80% of the (net, head) cases, and removing both gives a smaller sign share than the mean of the two single removals in
  >= 75% of the A nets.
- P258 (one head, nothing to cancel): in the B nets of ctrl and fix, removing the single pooling head's signed term changes the sign
  share by less than a factor 2 (ratio in [0.5, 2]) in >= 75% of them.
- P259 (membership frees the norm route): the median KL factor of a single-head removal in own is below ctrl's.
**Also reported (descriptive):** rom for base and all-zeroed; KL factors and sign-share ratios by cohort and type; per-net tables.
**What the outcomes would mean.**
- P256 holds: the learned pools' reliability computation runs largely through the signed term (sign-dependent within a head), not
  only through LayerNorm's norm; fails: the signed term is a minor part.
- P257 holds and P258 holds: an A net's two opposite-sign heads cancel much of each other's sign dependence, which a single head (B)
  cannot do; with E27 this accounts for A's smaller sign share. P257 fails: the pair does not act as a cancellation.
- P259 holds: with membership given the nets lean less on the signed term (their norm term raises large errors, E63); fails: they
  use it as much.
- These are evaluation-time surgeries that push the nets off their trained operating point; large KL factors measure how much the
  term is needed, not a mechanism of training.


## Amendment 75 (2026-10-31 05:24Z): E67, the head-level account in world 2, where nothing has to be routed: the query's block-1 pooling logits (CPU, existing nets)

Written by the seventh agent before any E67 quantity is computed on a registered net. Known beforehand: E1-E66 with their reviews
(E66: Update 75), in particular E63 and E65 (in world 1 every pooling head of the standard nets, which must find a source's tokens by
their codes, has a large positive norm-scaled own logit, median 16.8, and a norm term of about -3.5 nats at |r| = 3; nets told the
membership have small negative ones), E66 (the world-1 pooling relies heavily on the signed term), E40, E48, E54 (world 2's nets depend
on per-example signs; supplying the products halves the KL at D = 2; no consistent big-miss counterpart there).
**Exploration (disclosed; not results; on E40's d2ctrl s0, d2ctrl s1 and d1ctrl s0, excluded from every test below).** The query
token's block-1 heads put 0.79-0.95 of their mass on the examples; their norm-scaled logits L are -1.19 to +0.56; the norm term at an
example with |y| = 3 is -0.09 to +0.48; the signed coefficients have RMS up to 1.2 (y) and 1.7 (x); removing one head's signed terms
from the query row multiplies the KL by 2.2-12, both heads' by 5-20. Plumbing: the surgery with no head removed reproduces the forward
exactly (a ctrl net, and an sgnprod net at step 0); `e67_run.py` on the explored nets reproduces the exploration's values.
**Link to M2 (across tasks; what the architecture predicts).** In world 2 every context token is an example of the one task, so no
attention has to find a group's members. If the large positive norm-scaled logits of world 1's pooling heads are a routing requirement
(E63), world 2's pooling logits should lack them, and with them the negative norm term that lowers large errors' scores; if the signed
term is how a pre-LN head builds an even statistic from signed tokens (E66), world 2's pooling should rely on it too, and less where the
even statistics are supplied (sgnprod, E48). That would carry the head-level account to a second task with a different sufficient
statistic, and fit E54 (no consistent departure in world 2).
**Design** (`e67_run.py`, `e67_summ.py`; `t2_world.py`, `t2lin.py`, `e40_eval.py`, `pfn2.py` unchanged). Cohorts: E40's ctrl and canon nets
and E48's sgnprod nets at D = 1 and 2 (acquired, registered gap < 0.10), the three explored nets excluded. Per net on 2,000 episodes
(seed 67,001), for the query's two block-1 heads: attention mass; the exact logit decomposition over example keys (the signed raw
columns against the rest: the bias, and in sgnprod the products): L, the norm term at |y| = 3, the signed coefficients K (y, x); the
KL with the signed terms removed from the query row in one head and in both (as E66). One job per cohort in the analysis queue.
**Predictions.**
- P260 (no routing, no large routing logit): over the query heads of all registered nets, the median |L| is < 3 (world 1: 16.8).
- P261 (no routing-induced down-weighting): the norm term at |y| = 3 is > -1 in >= 80% of those heads.
- P262 (the signed route carries world 2's pooling too): in the ctrl nets, removing one head's signed terms multiplies the KL by >= 2
  in >= 75% of the (net, head) cases.
- P263 (supplying the even statistics lowers that reliance): the median single-head KL factor of the sgnprod nets is below the ctrl
  nets'.
**Also reported (descriptive):** by cohort, the query's mass on the examples, K RMS, the sign pairing of K_y across heads, KL factors for
both heads.
**What the outcomes would mean.**
- P260 and P261 hold: the large positive norm-scaled logits and the negative norm term are specific to attention that has to route,
  absent where it does not; with E54 this fits world 2's small, format-dependent departures. Fail: they are a general property of
  these nets' pooling, not of routing.
- P262 holds: pooling signed tokens relies on the signed term in both tasks; P263 holds: supplying the products reduces that reliance,
  as |r| did in world 1 (E65). Fail: world 2 pools differently.


## Amendment 76 (2026-10-31 06:00Z): E68, how the A allocation forms in training: when a lineage's second summary head starts pooling, and with which sign (CPU, checkpoint weights)

Written by the seventh agent before any E68 quantity is computed on a registered lineage. Known beforehand: E1-E67 with their reviews;
in particular E63's traj cohort (E37's 11 ctrl runs: every run has a summaries head by 2,000 steps; the A runs' second pooling head
appears at 5,000 (s1), 10,000 (s0, s2) or 20,000 steps (s6); no B run ever has two pooling heads at the 10 checkpoints), E63's
opposite-sign pairing of A's two pooling heads at the end of training, E66 (A's low sign share needs both heads' signed terms), E47
(the types separate by block-1 cross-source attention and the sign share at 10,000 steps), E57 (the cap scale separates them later).
**Exploration (disclosed; not results).** `e68_run.py`'s plumbing on the five nets explored in amendment 71 (E43 newinit s0, s11: A;
s1, s2, s4: B; excluded below), 200 bases: both A lineages have two pooling heads from 5,000 steps (P 0.79 and 0.73; 0.74 and 0.70),
with opposite-sign K there; the pair's |K| ratio (smaller over larger) is about 0.59 at that step and 0.81 at the end; the B lineages
pool with one head from 2,000-5,000 steps and never with two; every lineage's first pooling head keeps its K sign to the end.
`e68_summ.py` ran end to end on that output.
**Link to M2 (how they learn).** E63-E66 describe the end state: A pools with an opposite-sign pair of heads, B with one head. How
training reaches A decides which solution, and with it how much sign dependence and departure, a net ends with. If the second head,
when it starts pooling, already carries the sign opposite to the first's, the pair forms as a pair (it does not become a pair later),
which fits the reading that it forms because it cancels the first head's sign dependence; the timing locates the type decision.
**Design** (`e68_run.py`, `e68_summ.py`; `e63_run.py`'s functions, `e47_summ.py`'s lineage rules unchanged). Lineages: E47's (E37's ctrl
runs, E43's branches and new initialisations, E44's hybrids, E46's perturbed runs), acquired at the end (registered gap < 0.10), final
type from the registered middle-range rom sign, E43 brK lineages used only at steps > K; the five explored nets excluded. At the 10
checkpoints (0-90,000 steps), on E27's first 1,000 bases: P_h, C_h, and every head's mean L_own and mean K over the source's own
labelled keys. Pooling head: P_h >= 0.5. Jobs in the analysis queue (nice 19).
**Predictions.**
- P264 (the pair forms as a pair): among the final-A lineages with two pooling heads at some checkpoint, the two heads' mean K have
  opposite signs at the first such checkpoint in >= 80% of them.
- P265 (when): that first two-pooling-head checkpoint is at 5,000-20,000 steps in >= 70% of the final-A lineages.
- P266 (B never pairs): in >= 80% of the final-B lineages no checkpoint from 5,000 steps on has two pooling heads.
**Also reported (descriptive):** the distribution of the first two-pooling-head step; the pair's |K| ratio then and at the end; whether
the first pooling head keeps its K sign; pooling heads' L_own and the median C_max over training by final type.
**What the outcomes would mean.**
- P264 holds: the second head's signed coefficient starts opposite to the first's; the pair forms with the cancellation built in, as
  the reading that the pair forms because it cancels the sign dependence would have it (association in time, not a causal test).
  Fails: the pair starts same-signed or unsigned and the opposite signs appear later.
- P265 holds: the type is decided in the window in which E47's routing markers separate the types; fails: earlier or later.
- P266 holds: B lineages do not pass through the A allocation; fails: B lineages form a pair and lose it.


## Amendment 77 (2026-10-31 06:28Z): E69, does the head budget make the B type? Four block-1 heads of the same total width, same initial values and episodes (architecture; one L4, coordinator-approved, cap $1)

Written by the seventh agent before any E69 training (the smoke run below excepted). Known beforehand: E1-E68 with their reviews (E68:
Update 77), in particular: E63 and E65 (A nets pool a source's errors with both block-1 summary heads, an opposite-sign pair; B nets
with one, the other head with random codes on the three summaries; 29 of 30 and 222 of 222 nets), E66 (A's low sign share needs both
heads' signed terms), E68 (in every A lineage the pair appears as an opposite-sign pair when the second head turns, mostly at
5,000-20,000 steps; no B lineage pairs), E65 (in the sign-free format one pooling head suffices and the summaries head stays), E22's
older cohort (all 18 one-head nets were B type). E37's ctrl nets: 4 A, 7 B, one stall (seed 3); sign share 0.200; mean KL 0.0127.
**The idea.** With two block-1 heads, a net cannot both keep a head on the summaries (a consensus over the readings, which Bayes uses
through the query readings' marginal) and pool its source's errors with an opposite-sign pair; B keeps the summaries head, A gives it
up. With four heads of dimension 8 (the same 32 attention dimensions, the same qkv and output maps), both are possible, so if the head
budget is what makes B, most nets should end A, keep a summaries head, and pair their pooling heads.
**Plumbing (disclosed; not results).** `train_h4.py` builds pfn2's baseline with 4 heads per block; with seed 0 every parameter equals
E37 ctrl s0's step-0 checkpoint exactly (the qkv and output maps keep their shapes), and the episode stream's config name is the 2-head
one, so the episodes are ctrl's. A 1,000-step CPU smoke run of seed 0 (the registered run's first 1,000 steps; its validation gap is
printed by the run and seen) checks that it trains.
**Link to M2 (explain the inductive bias; predict from the architecture).** The head-level account (E63-E68) says the B type is a
budget choice between two jobs in a two-head first layer. Changing only the number of heads at a fixed width tests that causally: the
account predicts the type distribution, the head roles, the sign dependence and the KL of the 4-head nets before training.
**Design** (`train_h4.py`, `e69_heads.py`, `e69_summ.py`; `train_sw.py`, `e30_fmt.py`, `e48_fmt.py`, `e48_eval_w1.py`, `e37_eval.py`,
`e30_eval.py`, `e63_run.py`, `pfn2.py`, `world.py` unchanged; GPU scripts `gpu/gpu_jobs_x.sh`, `gpu/launch_x.sh`, `gpu/retry_x.sh`).
- Arm h4: 4 heads per block (both blocks), ctrl format, amendment 45's fixed 90,000-step schedule, sparse checkpoints, seeds 0-11,
  the same initial values and episodes as E37's ctrl nets. 12 trainings on one on-demand L4 (8 at once, two waves; estimated about
  1 h, $0.9; the VM is deleted by the runner, checked in the instance list and logged in GPU-LOG.md). E37's ctrl nets were trained on
  L4 too.
- Measures (CPU, after the run): `e48_eval_w1.py` (test KL, rom large and middle, type, block-2 lab, xq, the sign-flip cost) and
  `e69_heads.py` (E63's head measures for the four heads: pooling heads P_h >= 0.5, consensus heads C_h >= 0.5, L_own and K per head,
  an opposite-sign pair among the pooling heads). Acquired: gap < 0.10. Comparisons with E37's ctrl nets of the same seeds.
**Predictions** (the head budget makes the B type).
- P267: >= 75% of the acquired h4 nets are A-type (middle-range rom < 0).
- P268: in >= 75% of the acquired h4 nets, block 1's summary rows have at least two pooling heads and at least one consensus head.
- P269: the pooled sign share of the acquired h4 nets is <= 0.5 x ctrl's (0.200).
- P270: the paired mean KL ratio h4 / ctrl is <= 0.75.
**Also reported (descriptive):** the number of pooling and consensus heads per net, opposite-sign pairs among pooling heads, L_own and
K per head, rom, xq, lab, stalls, per-seed types against ctrl's.
**What the outcomes would mean (including refutations).**
- P267 and P268 hold: with a larger head budget the nets keep the summaries head and pool with a pair, and end A: the B type is a
  budget choice between the two jobs, as the account says. With P269 and P270 the budget also accounts for B's extra sign dependence
  and KL.
- Most h4 nets still end B (P267 fails):
  - with at least two pooling heads but no opposite-sign pair among them, or with the pair but still B: having enough heads for a
    pair is not sufficient, so the type is not decided by the head budget (it would be decided, for example, by the first pooling
    head's development or by the initialisation, E44), and the allocation seen in two-head nets is a consequence of the type;
  - with only one pooling head although four heads are available (the others on the summaries or elsewhere): the extra heads are not
    used for pooling, so the allocation is an attractor of training rather than a budget limit, and the account's "budget" reading
    fails;
  - in either case the account's claim that B is a budget choice is refuted for this recipe; the descriptive links of E63-E68 stand.
- P267 holds but P268 fails (A without a summaries head, or with a single pooling head): the type changes for another reason (for
  example the smaller head dimension), not through the predicted roles.
- P269 or P270 fails while P267 holds: the budget changes the type but not the sign dependence or the KL the account attributes to it.
- Smaller heads (dimension 8) are a confound for any outcome; stalls are reported.


## Amendment 78 (2026-10-31 07:40Z): E70, two controls: four heads in block 1 only (b1h4), and the missing fourth cell, membership given with sign-free inputs (ownmag) (architecture; one L4, coordinator-approved, cap $1.6)

Written by the seventh agent before any E70 training. Known beforehand: E1-E69 with their reviews (E69: Update 79: with four heads in
both blocks all 12 nets end A, with two opposite-sign pooling heads and a summaries head; the reviewer noted that both blocks change and
the head width halves). The 2x2 cells known so far (same seeds, initial values and episodes; test-NLL gap, acquired nets): ctrl (content
routing, signed) mean 0.0127, median 0.0104 (11 nets); own (membership given, signed; E60, CPU) 0.0042; mag (content routing,
sign-free; E37) mean 0.0055, median 0.0040 (12, one B net at 0.0221).
Plumbing (disclosed; not results): `train_b1h4.py` builds 4 heads (dimension 8) in block 1 and 2 heads (dimension 16) in block 2; with seed 0
every parameter equals E37 ctrl s0's step-0 checkpoint and the stream's config name is ctrl's. ownmag is `train_own.py --variant own`
in the mag format (no parameter added; the initial values and episodes are those of ctrl, mag and own).
**Link to M2.** E69's all-A result could come from block 2's extra heads or from narrower heads everywhere; b1h4 changes only block 1's
head count. The non-additivity reading of the KL excess (Update 78) lacked the fourth cell; ownmag supplies it.
**Design** (`train_b1h4.py`, `gpu/jobs_e70.txt`, `e70_summ.py`, `e69_heads.py`; `train_own.py`, `pfn_own.py`, `train_sw.py`, `e30_fmt.py`, `e48_fmt.py`,
`e48_eval_w1.py`, `e60_eval.py`, `e63_run.py`, `pfn2.py`, `world.py` unchanged; GPU scripts `gpu/gpu_jobs_x.sh`, `gpu/launch_x.sh`, `gpu/retry_x.sh`).
Arms (world 1, amendment 45's fixed 90,000-step schedule, sparse checkpoints, seeds 0-11): b1h4 (ctrl format) and ownmag (mag format).
24 trainings on one on-demand L4 (8 at once; estimated 1.5 h, $1.3); the VM is deleted by the runner, checked and logged. Evaluation on
CPU: `e48_eval_w1.py` for b1h4, `e60_eval.py` for ownmag (test KL, rom, type, lab, xq; sign share for b1h4); `e69_heads.py` for b1h4's
block-1 heads. Acquired: gap < 0.10. Comparisons on the same seeds: E37's ctrl and mag (GPU), E60's own (CPU).
**Predictions.**
- P271 (block-1 budget): >= 75% of the acquired b1h4 nets are A-type.
- P272: >= 75% of the acquired b1h4 nets have at least two pooling heads and at least one consensus head in block 1.
- P273 (redundant costs): over seeds acquired in the compared cells, the paired mean-KL ratios ownmag / own and ownmag / mag both have
  lower 95% bootstrap limits >= 0.80: once either change is made, the other removes little more.
- P274: mean rom_large of ownmag <= 0.03.
**The additive and non-additive decision rules for the fourth cell (both reported).** (1) Additive extrapolation: KL_add,s = KL_own,s +
KL_mag,s - KL_ctrl,s per seed (seeds acquired in all four cells), its mean and 95% bootstrap interval; if ownmag's mean KL lies above
the interval's upper limit, the costs are not additive on the KL scale (expected from the KL floor alone, so weak evidence). (2) The
informative rule is P273: "redundant" if both ratio lower limits are >= 0.80; "partly separate" if either ratio's upper limit is
< 0.80 (each change leaves a cost the other removes); otherwise undecided. Also reported: the multiplicative interaction
log KL_ctrl - log KL_own - log KL_mag + log KL_ownmag (mean over seeds, interval).
**What the outcomes would mean (including refutations).**
- P271 and P272 hold: four heads in block 1 alone suffice for the all-A outcome, locating E69's effect in block 1's head budget, as the
  account says. Most b1h4 nets B, or with fewer than two pooling heads: E69's result needed block 2's extra heads or narrower heads
  in both blocks; block-1 head scarcity is then not the cause, and the account's budget reading is refuted at block 1.
- P273 holds: membership and sign-free inputs remove the same cost (redundant), as the head-level account says (the excess needs content
  routing and signed evidence together). "Partly separate": each removes a part the other does not; the excess is then partly two
  costs. P274 tests whether learned pooling with membership and |r| removes the departure.


## Amendment 79 (2026-10-31 11:03Z): E71, head count or head width? Block 1 with two heads of dimension 8, or three or four heads of dimension 16 (architecture; trained on CPU)

Written by the eighth agent before any E71 training (the plumbing below excepted). Known beforehand: E1-E70 and E64 with their
reviews (E64: Update 81). In particular: E63, E65 and E68 (A nets pool a source's errors with both block-1 summary heads, an
opposite-sign pair; B nets with one, the other head with random codes on the three summaries); E69 (four heads of dimension 8 in
both blocks: 12 of 12 A) and E70 (four heads of dimension 8 in block 1 only, b1h4: 12 of 12 A, two pooling heads plus a consensus
head in all 12, sign share 0.067, paired KL ratio 0.46 to ctrl); E64 (block 1's summaries barred from each other: 11 of 12 A). The
reviews of E69 and E70 noted that head count and head width changed together. E37's ctrl (2 heads of dimension 16): 4 A, 7 B, one
stall (seed 3); sign share 0.200; mean KL 0.0127; first checkpoint with two pooling heads in ctrl-family A lineages (E68): median
10,000 steps. E60: four ctrl seeds retrained on CPU matched their GPU twins (KL within 4%, rom within 0.002, the same types).
**The idea.** The head-level account reads the B type as a budget choice: with signed evidence and the summaries reachable, block 1's
summary rows need three attention patterns (a summaries head and two pooling heads with opposite signed coefficients), and a
two-head block 1 must drop one job (A drops the summaries head, B a pooling head). This count reading predicts the type mix from
the number of block-1 heads alone, with a threshold at three heads, and no effect of the head dimension in this range. A width
reading: narrower heads make A regardless of their number (for example, because a head of dimension 8 routes the 16-dimensional
source codes through a rank-8 query-key map, so one head cannot route precisely and two heads share the pooling). E69 and E70
changed both at once (2 x 16 to 4 x 8).
**Plumbing (disclosed; not results).** `pfn_b1hw.py` builds block 1 with H1 heads of dimension DH1 (an attention inner dimension
H1 DH1 separate from the width 32) and block 2 with pfn2's 2 heads of dimension 16; `paired_init` copies the ctrl initialisation
of the same seed: every parameter of the same shape; for q, k and v, the rows of ctrl's heads 0 and 1 in their first DH1
dimensions; heads 2 and 3 drawn from the default U(-1/sqrt(32), 1/sqrt(32)) bound with torch.Generator seeded 71000 + 16 seed + h;
block 1's output-map columns and bias from ctrl's, rescaled by sqrt(32 / (H1 DH1)) to the default bound for the new fan-in (new
heads' columns drawn at that bound). So h2w8's q, k, v are E70 b1h4's heads 0 and 2, h3w16 is ctrl plus one head, and h4w16 is
h3w16 plus one head. `e71_plumb.py` (initial weights only): ctrl's initialisation as train_sw builds it equals E37 ctrl's step-0
checkpoint (seeds 0, 5); the copied parts and the nesting hold for all 12 seeds; at initialisation, on 512 episodes, block 1's
summary-row logit standard deviation is 0.28-0.39 in every cell, the own-token attention mass 0.27 (uniform), and the ratio of the
attention output's norm to the residual's 0.89-1.19 (ctrl 0.92-1.19). Smoke runs of 1,000 steps (const 900, cooldown 100) for
h4w16 and h2w8 at seed 99, outside the registered seeds, checked training, meta.json and both loaders (their printed validation
gaps at 1,000 steps were 0.18-0.19); `e71_summ.py` was run on the existing ctrl and b1h4 cells only (it reproduces E70's values)
and its outputs deleted.
**Link to M2 (explain the inductive bias; predict the departure from the architecture).** The account's central claim is that the
architectural quantity behind the B type is the number of block-1 attention patterns relative to the jobs the task needs. This
episode separates that quantity from the head dimension and tests the predicted threshold (three heads) before training.
**Design** (`pfn_b1hw.py`, `train_b1hw.py`, `e71_eval.py`, `e71_heads.py`, `e71_queue.py`, `e71_loop.sh`, `e71_summ.py`,
`e71_plumb.py`; `train_sw.py`, `e30_fmt.py`, `e48_fmt.py`, `e48_eval_w1.py`, `e37_eval.py`, `e30_eval.py`, `e63_run.py`, `pfn2.py`,
`world.py`, `dynq.sh` unchanged).
- Arms (world 1, ctrl format, train_sw's fixed 90,000-step schedule, sparse checkpoints, seeds 0-11, `--heads 2` so the episode
  stream's config name is ctrl's and the episodes are ctrl's): h2w8 (block 1: 2 heads of dimension 8), h3w16 (3 of 16), h4w16 (4 of
  16). With ctrl (2 x 16, E37) and b1h4 (4 x 8, E70) this gives the head count at a fixed dimension (16: 2, 3, 4; 8: 2, 4) and the
  dimension at a fixed count (2 heads: 8, 16; 4 heads: 8, 16).
- 36 trainings on General's CPU (`dynq.sh /tmp/s7t`, 3 at once, nice 19, ionice 3; about 6.5 h): h2w8 and h4w16 interleaved by
  seed first, then h3w16. ctrl and b1h4 were trained on L4 GPUs.
- Measures (CPU): `e71_eval.py` (e48_eval_w1's test KL, rom_large, rom_middle, type, block-2 attention to the labelled tokens lab_b2,
  xq, the flip_ind sign-flip cost); `e71_heads.py` (E63's block-1 summary-head measures: pooling heads P_h >= 0.5, consensus heads
  C_h >= 0.5, L_own and K per head, an opposite-sign pair among the pooling heads) on model.pt and at the checkpoints 0, 2k, 3k, 5k,
  10k, 20k, 40k, 60k, 80k, 90k, for E71's nets and for E70's 12 b1h4 nets (their trajectories have not been measured before).
  Acquired: gap < 0.10; type A: rom_middle < 0. Paired ratios over the seeds acquired in both cells, 2,000 bootstrap resamples of
  seeds (numpy seed 710).
**Predictions** (the count reading).
- P275 (four heads suffice at dimension 16): >= 75% of the acquired h4w16 nets are A-type, and >= 75% have >= 2 pooling heads and
  >= 1 consensus head in block 1.
- P276 (narrow heads do not suffice at two heads): <= 50% of the acquired h2w8 nets are A-type (ctrl: 4 of 11).
- P277 (the threshold is three heads): as P275 for h3w16 (with three heads: two pooling heads and one consensus head).
- P278: for h4w16 and for h3w16, the sign share is <= 0.100 (half of ctrl's) and the paired mean-KL ratio to ctrl is <= 0.75 (point
  estimates).
- P279 (the count matters at dimension 8): the paired mean-KL ratio h2w8 / b1h4 has a lower 95% limit >= 1.25.
- P280 (with spare heads the pair forms early): over the A lineages of b1h4, h3w16 and h4w16 together, the median first checkpoint
  with >= 2 pooling heads is <= 5,000 steps (ctrl-family A lineages, E68: 10,000).
For P275-P277, an arm with fewer than 6 acquired nets is reported as undecided.
**Also reported (descriptive):** the per-seed types against ctrl's and b1h4's; per cell the stalls, mean and median KL, sign share,
rom_large and rom_middle, lab_b2, xq, the pooling and consensus head counts, opposite-sign pairs, the pooling heads' L_own; the paired
log-KL contrasts for the count (h4w16 against ctrl; b1h4 against h2w8; h3w16 against ctrl) and for the dimension (h2w8 against
ctrl; b1h4 against h4w16); the trajectories (first checkpoint with two pooling heads, their signs there, the consensus heads over
training) per cell.
**What the outcomes would mean (including refutations).**
- P275 and P276 hold: the type follows the block-1 head count, not the head dimension, in this range: the claim is the head budget.
- P275 holds and P276 fails (two narrow heads mostly A): more heads and narrower heads each lead to A; the budget reading is then one
  of two routes, and the width route needs its own account.
- P275 fails (four heads of 16 mostly B) and P276 fails: the dimension, not the count, made E69's and E70's all-A outcome; the
  budget reading is refuted at block 1.
- P275 fails and P276 holds: neither alone suffices; the all-A outcome needs four narrow heads (or four heads within 32 attention
  dimensions), and the budget reading fails as stated.
- P277 holds: three heads suffice, the threshold the count of three jobs predicts. P277 fails while P275 holds: more than three
  heads are needed, so the account's job count is incomplete (E69 found two consensus heads in 4 of 12 nets).
- P278 holds: the sign dependence and the KL follow the count as they followed E69's and E70's change; fails: the count changes the
  type but not the costs the account attributes to it.
- P279 holds: at dimension 8 the count changes the KL as the account says; fails: at dimension 8 the count barely matters.
- P280 holds: with spare heads the pair forms when the first pooling head does, consistent with the two-head nets' later second
  pooling head (5,000-20,000 steps) being a competition with the summaries job; fails: the pair's timing is set by something other
  than that competition (for example the slow growth of the signed term).
- Confounds for any outcome: the total attention dimension and the parameter count change with the arms (h2w8: 16 dimensions,
  16,636 parameters; h3w16: 48; h4w16: 64, 22,924; ctrl and b1h4: 32, 18,732), as does the output map's initial scale per column;
  E71 trains on CPU, ctrl and b1h4 on GPU (E60's check); stalls are reported.


## Amendment 80 (2026-10-31 11:41Z): E72, is the big-miss departure the information loss of block 1's own-source pooled statistic? Constrained-Bayes readers of four reductions of the pool, and route patching (CPU, existing nets)

Written by the eighth agent before any E72 quantity on a held-out net (the exploration below used dev nets only). Known beforehand:
E1-E70 and E64 with their reviews and amendment 79 (E71, in flight); in particular E12-E13 (the departure, rom on E12's matched
pairs), E25 (older cohort: matching the source's own pooling terms at its summary token removes the contrast), E34-E36 (B nets'
block 2 reads the labelled tokens; path patching), E53, E55, E58, E61 (uniform pools: departure close to constrained Bayes, CB, of
the capped mean at the net's cap scale c), E55 (learned pools: about three times that value at their c), E63-E68 (pooling heads,
L_own, K), and the seventh agent's note that `e68_rpm.py` underpredicts A nets and overpredicts B nets on four explored nets.
**Exploration (disclosed; dev nets only, all excluded from every test).** E43's newinit nets s0, s1, s2, s4, s11 (amendment 71's
explored set; s0 and s11 A-type, s1, s2, s4 B-type). Scripts `e72x_explore.py`, `e72x_ladder.py`, `e72x_patch.py`; outputs
`results/e72x_dev.json`, `results/e72x_ladder.json`, `results/e72x_patch.json`, `results/e72x_rpmcheck.log`.
- The reader pipeline: the sufficient statistic through the classifier gives rom 0.000 / 0.000; the uniform capped mean at c = 1.4
  gives 0.166 / 0.103 (amendment 63's table: 0.165 / 0.102).
- CB rom (large / middle) against the nets': s0 (net 0.082 / -0.041): M0 0.031 / +0.013, M1 0.017 / -0.029, M2 0.091 / +0.011,
  M3 0.074 / -0.033; s11 (0.111 / -0.030): 0.027 / +0.012, 0.035 / -0.024, 0.095 / +0.014, 0.088 / -0.033; the B nets s1, s2, s4
  (0.133-0.137 / +0.061-0.074): M0 about 0.04 / +0.016, M1 0.33-0.35 / +0.17-0.19, M2 0.35-0.38 / +0.15-0.16, M3 0.43-0.45 /
  +0.11-0.13. The statistic's dependence on the query reading is small (RMS change 0.03-0.06 of its standard deviation per unit o*).
- Logit noise does not explain M1's shortfall: L_j varies by 0.013-0.037 nats within a source (1.5-2.0 between sources) and K_j by
  0.5% within a source; adding this noise to M1 changed its rom by at most 0.02 (M1w, M1bw, M1bwK). The remaining differences
  between M1 and M2 (the real per-token norm with rho and A_j, and the real weights' other details) were not isolated.
- Route patching (each patch minus its detach-only control): A nets: c_i 0.090 and 0.115 (nets 0.082, 0.111), |c_ii| and |c_iii| at
  most 0.005; B nets: c_i 0.233-0.248 (nets 0.133-0.137), c_iii -0.039 to -0.081, c_ii at most 0.004. The detach-only control of
  route iii moves the big-miss weight by -0.17 to -0.18 of Bayes's weight in B nets (A: -0.004, -0.012). The unpatched rom
  recomputed by the patch code equals the registered evaluation's in all five nets.
**Plumbing (disclosed; not results).** `e72_run.py` and `e72_patch.py` reproduce the exploration's values on s0 (M0-M3 and every
route contribution to all printed digits) and s1 (routes). With step-0 weights (E72_NSIM = 6,000): the uniform-pool variant's
summary rows put exactly 1/4 on each own labelled token (P_h = 1.0 in both heads), the membership-given variant's P_h = 1.0, and the
width-128 PCA path runs (E72_FORCEPOOL, plumbing only).
**The idea.** (1) In A nets the departure is information loss: a Bayes reader of the real own-source pooled values (M3) reproduces
both its size and its middle-range sign, the summary's own block-1 pool is the route that carries it, and block 2's direct read
of the labelled tokens carries nothing: the downstream computation reads the pool nearly optimally. (2) In B nets the pool alone
would give about three times the departure, the nets' own reading of that pool gives about 1.8 times it, and block 2's direct read
of the labelled tokens lowers it: B's departure is not its pool's information loss; B nets read the pool less sharply than a Bayes
reader and partly correct through block 2. (3) The idealised reduced model (mean logit coefficients, rho = 0, one cap scale)
misses most of the large-range departure of content-routing A nets.
**Link to M2 (predict the departure).** The departure is the one quantity the head-level account had not predicted in size. This
episode tests, across held-out architectures (content routing, membership given, barred summaries, four heads, fixed codes),
formats (signed, sign-free) and widths (32, 64, 128), whether its size and sign follow from block 1's computation through a Bayes
reader, and where the two types differ.
**Design** (`e72_run.py`, `e72_patch.py`, `e72_queue.py`, `e72_summ.py`; `e68_rpm.py`, `e53_cb.py`, `e63_run.py`, `lngeom.py`,
`pfn_b1hw.py`, `e30_eval.py`, `e30_fmt.py`, `e48_fmt.py`, `pfn_own.py`, `pfn_uni.py`, `pfn_e64.py`, `pfn2.py`, `world.py` unchanged).
- Part 1 (`e72_run.py`, every acquired net): M0-M3 as in its docstring (pooling heads P_h >= 0.5 under the net's own block-1
  attention; the uniform-pool and membership-given variants pool with every head by construction).
- Part 2 (`e72_patch.py`, nets with pfn2's standard forward): route contributions c_i, c_ii, c_iii and detach controls.
- Held-out cohorts (registered): ctrl (E37, 11 acquired), fix (E30, 12), own (E60, 12), nocons (E64, 12), h4 (E69, 12), b1h4 (E70,
  12), mag (E37, 12), ownmag (E70, 12), w64ctrl, w128ctrl, w64mag, w128mag (E38, 10 each); calibration: uni (E53, 12); descriptive:
  d3ctrl, d3mag (E38, depth 3, where later blocks can re-pool); secondary, analysed with the same tests once their evaluations
  exist: E71's h2w8, h3w16, h4w16. Part 2 for ctrl, fix, h4, b1h4, mag and the four E38 width cohorts (d3 descriptive, E71
  secondary). Types: the registered middle-range rom sign; acquired: gap < 0.10.
- Compute: the analysis queue (`/tmp/s7q`, nice 19), one process while E71 trains, after E71's evaluation jobs; about 2.5 min per net.
**Predictions** (sets as in `e72_summ.py`: A = the A-type nets of the registered cohorts; routing-A = A outside own and ownmag; B =
the B-type nets of the learned-pool registered cohorts with net rom_large > 0).
- P281: |M3 - net| <= 0.03 for rom_large in >= 75% of A.
- P282: M3's rom_middle < 0 in >= 75% of A.
- P283: over the registered cohorts with >= 3 A nets (at least 8 cohorts), the Spearman correlation of cohort-mean M3 with
  cohort-mean net rom_large is >= 0.6.
- P284: M1's rom_large < net rom_large - 0.03 in >= 75% of routing-A.
- P285: M3's rom_large >= 1.5 x net rom_large in >= 75% of B.
- P286 (Part 2, A nets): |c_i - net| <= 0.03 and |c_iii| <= 0.02 (rom_large) in >= 75%.
- P287 (Part 2, B nets): c_i >= 1.3 x net and c_iii < 0 (rom_large) in >= 75%.
- P288 (calibration): |M3 - net| <= 0.03 for rom_large in >= 75% of the uni nets.
**Also reported (descriptive):** per cohort and type the mean net rom and M0-M3 (large, middle), M3 - net, c, pooling-head counts;
Part 2's median routes and detach effects; d3; the secondary set's tests.
**What the outcomes would mean (including refutations).**
- P281-P283 hold: across these architectures, formats and widths, the departure of A-type nets is the information loss of block
  1's own-source pool, read nearly optimally downstream: its size and sign are computed from block 1's pooled values by a Bayes
  reader. P281 or P283 fails: the own pool's information loss does not predict held-out A nets' departures (the dev agreement was
  specific); P282 fails: the middle-range sign is not information loss of the pool.
- P284 holds: the reduced model's idealisation loses most of the departure, which then depends on details of the real pool beyond
  three numbers per head; fails: the reduced model suffices on held-out nets.
- P285 and P287 hold: B nets' departure is not their pool's information loss: they read the pool less sharply than a Bayes reader and
  block 2's direct read partly corrects it. Either fails: B nets read their pool as A nets do, or the correction is not by block 2.
- P286 holds: in A nets the summary's own pool carries the departure (E25's site) and block 2's read adds nothing; fails: other
  routes carry part of it.
- P288 holds: M3's reader also reproduces the uniform-pool nets' departure (consistent with E55); fails: the reader is miscalibrated
  there (for example, its value features carry information the uniform-pool nets do not use).
- Limits for any outcome: a classifier estimate of each reader (pipeline error about 0.001 on known cases); M3's value vectors
  include code and label parts; the derivative holds T fixed; the reader factorises over sources; the cohorts share seeds and
  initial values (E37's) and are not independent replications; Part 2's patches leave the trained operating point only for one
  route at a time.


## Amendment 80, addendum 1 (2026-10-31 13:36Z): E72, the like-for-like departure: the reading's path through its own pool (CPU, existing nets)

Written by the eighth agent after E72's Part 2 (route patching) had finished on the held-out nets and had been tabulated (A nets: the
own pool carries the departure in 64 of 66; B nets: 17 of 33 meet P287), and before any held-out Part 1 (CB) result was looked at
(the Part 1 files of the first cohorts exist and have not been opened). Not a change to amendment 80's predictions or analyses.
**What was found (dev nets only; scripts `e73x_codes.py`, `e73x_surg.py`, `e72x_ladder2.py`, outputs `results/e73x_*.json`,
`results/e72x_ladder2.json`).** (1) On E43 newinit s0, s11 and s1, detaching only the summary rows' block-1 queries from the
autograd graph leaves every prediction unchanged (identical test KL) and raises rom_large from 0.082, 0.111, 0.133 to 0.123, 0.150,
0.181: the net's weight on a reading includes a path through the source's own pool (the reading is part of the summary row's
query, which sets the own-pool attention), and that path lowers the departure by 0.04-0.05. The CB readers hold their statistic
fixed, so their like-for-like comparison is the detached ("detq") rom, against which M3 (0.074, 0.088) is about 60%. (2) Idealising
the codes at evaluation leaves rom unchanged (within 0.01) and cuts the between-source spread of the pooling coefficients by only
10-20%. (3) Replacing each pooling row's own-key attention by the head's global-mean coefficients (keeping the row's total own mass)
raises rom and the KL (by about 50%) relative to the same surgery with source-specific coefficients: the source-to-source variation
of the coefficients is compensated downstream, not plain noise. (4) In a CB ladder on real tokens, the source-specific coefficients
(V_src) give the real weights' CB departure (0.092, 0.100) and the global-mean coefficients (V_coef) about half (0.042, 0.043).
**Design** (`e72_detq.py`; reused: `e63_run.py`, `e15_run.py`, `pfn2.py`, `pfn_uni.py`, `pfn_b1hw.py`). For every net of E72's
cohorts (registered, calibration, descriptive, secondary): rom on E12's pairs through the net's own forward and masks, base
(recomputed) and detq (the summary rows' block-1 queries detached; for the uniform-pool variant detq equals base by construction).
Plumbing (disclosed): on dev s0, base 0.0818 (registered 0.0818) and detq 0.1225 (the exploration's 0.1225); on step-0 weights of
uni s0, detq equals base exactly; the own variant's path runs. About 2 s per net, in the analysis queue.
**Predictions** (sets as in amendment 80).
- P289: rom_detq - rom_base > 0 (rom_large) in >= 75% of the A and B nets of the registered learned-pool cohorts (the reading's path
  through its own pool lowers the departure).
- P290 (A, like-for-like): M3's rom_large < rom_detq - 0.02 in >= 75% of A (the A nets' departure at a fixed pool exceeds a Bayes
  reader's of the same pool).
- P291 (B, like-for-like): M3's rom_large > rom_detq + 0.05 in >= 75% of B (the B nets' departure at a fixed pool falls short of a
  Bayes reader's).
**Also reported:** per cohort and type the decomposition rom_base = M3 + (rom_detq - M3) + (rom_base - rom_detq): the pool's
information loss read by a Bayes reader, the net's reading of its pool beyond that, and the reading's path through its own pool.
**What the outcomes would mean.** P289 holds: the reading modulates its own source's pool in a way that lowers the departure, a route
the CB readers do not model, so amendment 80's P281 (M3 against the total rom) compares unlike derivatives and must be read
together with P290. P290 and P291 hold: like-for-like, A nets over-weight a big-miss source beyond what a Bayes reader of their pool
would and B nets under-weight it; the departure then has three measured parts of different sign by type. P290 fails with M3 within
0.03 of rom_detq in most A nets: the A nets read their pool like a Bayes reader at a fixed pool (the dev gap was specific). P291
fails: B nets read their pool as a Bayes reader would at a fixed pool.


## Amendment 81 (2026-10-31 17:01Z): E73, how the departure's three parts form over training: does the A type shed its departure by improving the pool and the B type by reading around it? (CPU, checkpoint weights)

Written by the eighth agent before any E73 quantity on a registered lineage. Known beforehand: E1-E72 with their reviews (E72: Update
82), in particular E72's decomposition of the final nets' departure into M3 (a factorised constrained-Bayes benchmark of the source's
own block-1 pool), detq - M3 (the net at fixed summary-row attention beyond it) and base - detq (the summary rows' query dependence);
E16 (the departure appears with reliability weighting and then shrinks 3-5 fold), E35 (B's block-2 attention to the labelled tokens
rises after acquisition), E63 and E68 (A lineages pair their pooling heads at 5,000-20,000 steps; B lineages keep one pooling head).
**Exploration (disclosed; dev lineages only, excluded).** `e73x_traj.py` on E43's newinit s0, s11 (A) and s1, s2, s4 (B) at 2k, 3k,
5k, 10k, 20k, 40k and 90k steps (`results/e73x_traj.json`). A lineages: one pooling head at 3k with base 0.34-0.39 and M3 0.43; the
opposite-sign pair by 5k; from 5k to 90k base falls 0.31-0.38 to 0.08-0.10 and M3 0.27-0.29 to 0.07 (M3's fall 0.75-0.88 of base's),
detq - M3 stays +0.02 to +0.07 from 5k on, and base - detq moves from +0.01/+0.03 to -0.04. B lineages: one pooling head throughout;
from 5k to 90k base falls 0.32-0.43 to 0.13-0.14 while M3 stays at 0.43-0.47 (changes of 0-4%), detq - M3 falls from -0.01/-0.10 to
-0.25/-0.26, and base - detq from -0.004/-0.030 to -0.04/-0.06. The summary's tests (`e73_summ.py`) give the registered outcomes on
these dev series (P292, P296 in both A; P293, P294 in all three B; P295 in all five).
**The idea.** The departure's slow decay after acquisition (E16) happens by two different routes. In A lineages the pool itself gets
better at preserving a big miss (M3 falls with the departure) once the opposite-sign pair forms, and the net's excess over the
benchmark at fixed attention is present throughout; in B lineages the single pooling head's pool never improves (M3 flat), and the
departure falls because the net's reading moves away from that pool (detq - M3 falls), consistent with B's growing block-2 read of the
labelled tokens (E35). In both types the summary rows' query dependence grows into a negative term.
**Link to M2 (how they learn; why they deviate).** This tests, over training, whether the A/B types reduce their departure by
different mechanisms, which would explain why the B type keeps a larger departure (its pool does not improve) and connects the head
allocation (E63-E71) to the departure's dynamics.
**Design** (`e73_traj.py` = `e73x_traj.py` with its docstring, `e73_summ.py`; `e72_run.py`, `e72_detq.py`, `pfn_l2fix.py`, `e63_run.py`,
`e68_rpm.py`, `e53_cb.py`, `pfn2.py`, `world.py` unchanged). Lineages: E37's 11 acquired ctrl runs (final types from the registered
evaluations: A s0, s1, s2, s6; B s4, s5, s7, s8, s9, s10, s11) at 2k, 3k, 5k, 10k, 15k, 20k, 40k and 90k steps; secondary (A tests only):
E69's 12 h4 and E70's 12 b1h4 runs at 2k, 3k, 5k, 10k, 20k, 40k, 90k. Per checkpoint: base, detq, M1, M3 (100,000 simulated episodes),
pooling heads (forced to the head with the largest P_h when none reaches 0.5), the test-NLL gap. One job per lineage in the analysis
queue (nice 19), output `results/e73_traj/<tag>.json`; later jobs may run on a second queue once E71's trainings end.
**Predictions.**
- P292 (A: the pool improves): M3(5k) - M3(90k) >= 0.6 x (base(5k) - base(90k)), with base(5k) > base(90k), in >= 3 of the 4 A lineages.
- P293 (B: the pool does not improve): |M3(90k) - M3(5k)| <= 0.15 x M3(5k) in >= 75% of the B lineages with an unforced pooling head at
  5k and at 90k.
- P294 (B: the reading moves away from the pool): (detq - M3)(90k) <= -0.10 and <= (detq - M3)(5k) - 0.10 in >= 75% of those B lineages.
- P295 (the query dependence grows negative): (base - detq)(90k) < (base - detq)(5k) in >= 75% of the 11 lineages.
- P296 (A: the excess at fixed attention is present from 10k on): detq - M3 > 0 at every checkpoint from 10k on in >= 3 of 4 A lineages.
- Secondary: P292 and P296 in >= 75% of the h4 and of the b1h4 lineages (all A).
**What the outcomes would mean.** P292 and P293-P294 hold: the two types shed the departure by different routes (A by improving the
pool, B by reading around an unchanged pool), so B's larger final departure is the price of a pool that does not improve, the
dynamical counterpart of E72. P292 fails: in A lineages the decay is not mainly the pool's. P293 fails: B pools improve too. P294
fails: B's decay is not the reading's. P295 fails: the query-dependence term is not a product of later training. P296 fails: the A
excess appears only late. Secondary failures would limit the A pattern to the two-head architecture.
**Limits.** The benchmark's limits (E72's review: factorised over sources, classifier-estimated, statistic held fixed); early checkpoints
before acquisition (2k) are far from Bayes; forced pools at checkpoints without a pooling head; the primary lineages share E37's seeds
with E72's final nets; one recipe and width.


## Amendment 82 (2026-10-31 17:15Z): E74, does the cost of finding a source's tokens by content grow with the number of sources? Six sources, content routing against membership given (architecture; trained on CPU)

Written by the eighth agent before any E74 training (the smoke runs below excepted). Known beforehand: E1-E72 with their reviews and
amendments 79 (E71, in flight) and 81 (E73, in flight); in particular E60 and E62 (with three sources, membership given removes most of
ctrl's KL excess: 0.0042 against 0.0127, code dependence and run-to-run variation), E63 and E65 (the pooling heads' large norm-scaled
routing logit L_own, 14-22 at width 32, lowers large errors' scores; leakage to other sources' tokens grows with code overlap), E72 (in
A nets the departure follows the source's own pool). No net with more than three sources has been trained or examined in this lane.
**The idea.** In the standard architecture a summary row finds its source's four labelled tokens by matching random identity codes in
the attention that also weighs their errors. With six sources it must single them out among 20 other sources' labelled tokens and 6
summaries (against 8 and 3), with more code overlap among six random 16-dimensional codes. The account (E60-E64b) then predicts a larger
cost of content routing relative to membership given, more leakage, a larger routing logit and with it a stronger norm term and a
larger departure, while nets told the membership (no routing) keep a small departure.
**Link to M2 (holds across tasks and sizes; predicts the departure from the architecture).** The number of sources is a property of the
task that the account says should scale the routing cost; this is the first test of the account outside S = 3, and it connects to the
scale-free lane's result on routing with more sources.
**Plumbing (disclosed; not results).** `e74_data.py` builds, for S = 6 and S = 3 with the same construction, 16,000 test episodes with exact
Bayes NLL (`world_n.World(S).bayes`, NV^S components) and E12-style matched pairs (kappa 0.25 and 0.85 at equal SS, E12's middle and
large ranges) for 9,000 / S episodes (5,902 rows at S = 6, 6,011 at S = 3) with exact Bayes effective weights; the Bayes mean, variance
and weights are invariant across the versions to 1e-12. `e74_eval.py` (test KL, rom, type, block-1 summary-head measures for S sources)
ran on two 1,000-step smoke nets at S = 6, seed 99 (ctrl, own; outside the registered seeds), and, on an earlier draft of the S = 3 data
(8,000 test episodes, 2,948 pair rows; regenerated since), on E37 ctrl s0 and E60 own s0 (KL 0.0084 and 0.0078, rom_large 0.078 and
0.020, ctrl s0 with two opposite-sign pooling heads and leakage 0.009): a six-source ctrl training runs about 1,000 steps in 40 s.
**Design** (`e74_data.py`, `e74_eval.py`, `e74_queue.py`, `e74_loop.sh`, `e74_summ.py`; `train_n.py`, `world_n.py`, `pfn_n.py`, `pfn2.py`,
`pfn_own.py` unchanged).
- Arms (world 1 with S = 6, ctrl format, pfn2's baseline at width 32 with two heads, the fixed 90,000-step schedule, sparse checkpoints,
  seeds 0-11): ctrl (content routing) and own (E60's restricted learned pool: summary rows see only their source's labelled tokens).
  `train_n.py` starts every seed from the same initial values as the three-source ctrl net of that seed (the parameters do not depend on
  S); its episode stream is seeded by the config name with '_S6'. 24 trainings on General's CPU after E71's (`dynq.sh /tmp/s7t`, 3 at
  once, nice 19; about 9 h).
- References at S = 3: E37's acquired ctrl nets and E60's own nets, evaluated with the same code on the S = 3 data.
- Measures: `e74_eval.py` (test KL on 16,000 episodes; rom_large, rom_middle on the matched pairs; type A if rom_middle < 0; for ctrl,
  block 1's pooling and consensus heads, an opposite-sign pair, L_own, the norm term at |r| = 3, and the pooling heads' mass on other
  sources' labelled tokens). Acquired: gap < 0.10.
**Predictions** (`e74_summ.py`).
- P297 (routing cost grows): over the seeds acquired in all four cells, the ratio of the paired-mean KL ratios, (ctrl / own at S = 6) /
  (ctrl / own at S = 3), has a point estimate >= 1.25 and a lower 95% limit >= 1.0.
- P298: the median L_own of ctrl's pooling heads is at least 1 nat larger at S = 6 than at S = 3.
- P299: the pooling heads' mean mass on other sources' labelled tokens in ctrl is at least 1.5 times larger at S = 6.
- P300: ctrl's mean rom_large at S = 6 is at least 0.02 above ctrl's at S = 3.
- P301: own's mean |rom_large| at S = 6 is at most 0.05.
**Also reported (descriptive):** acquisition counts (stalls), KL and rom per cell, types, pooling / consensus heads and opposite pairs in
ctrl at S = 6 (whether the head allocation account's A/B split recurs), the norm term, per-seed tables.
**What the outcomes would mean.** P297-P300 hold: with more competing sources, finding a source's tokens by content costs more, leaks more,
needs a larger routing logit and makes a larger departure, while membership given stays near Bayes's weighting (P301): the routing
account scales with the task as predicted. P297 fails (the ratio does not grow): the routing cost does not scale with the number of
sources in this range, a limit of the account's routing reading; P298-P299 failing with P297 holding would put the cost elsewhere than
in the routing logit or leakage; P300 failing would separate the departure from the routing cost; P301 failing would show a departure
without routing at S = 6. Many stalls at S = 6 would be reported as a result (harder acquisition) and limit the comparisons.
**Limits.** One width and recipe; the S = 3 references were trained on GPU (E37) and CPU (E60), the S = 6 nets on CPU (E60's check:
CPU and GPU agree on KL and rom); Bayes itself is sharper at S = 6, so the ratio, not the KL, is compared; the S = 6 episode stream differs.


## Amendment 83 (2026-10-31 20:25Z): E75, does a source's own reading, through the block-1 query of its summary row, carry information the nets use? Shuffling the reading in the pooling or the consensus heads' queries at evaluation (CPU, existing nets)

Written by the eighth agent before any E75 quantity on a held-out net. Known beforehand: E1-E73 with their reviews and amendment 82 (E74,
in flight); in particular E72's addendum 1 and E73 (detaching the summary rows' block-1 queries from the derivative raises the
departure in 126 of 135 nets; the term grows over training), E63-E68 (pooling and consensus heads; the signed coefficient K).
**Exploration (disclosed; dev nets only, excluded).** `e75x_qpath.py` on E43's newinit s0, s11 (A) and s1, s2, s4 (B)
(`results/e75x_qpath.json`, `results/e75x_qpath_split.json`): replacing the reading in every summary-row query by 0 or by another
episode's reading raised the test-NLL gap 1.02-1.06 fold in the A nets and 1.5-2.7 fold in the B nets, and rom_large by 0.03-0.08 in
all. Restricted to head types: in the B nets the pooling head's query carries it (shuffled: 2.0-2.7 fold, rom +0.04-0.06) and the
consensus head's does not (1.01-1.03, rom unchanged); in the A nets shuffling both pooling heads' queries gives 1.02-1.06 fold and rom
+0.034. The shuffled reading barely changes the attention itself (own-token mass unchanged; within-own total variation 0.015 in B, 0.008
in A). The B pooling heads' source-mean signed coefficient K correlates with the reading (-0.37, -0.38, -0.42; slopes -0.07 to -0.10 per
unit reading against |K| 1.8-2.0), the A heads' does not (-0.05 to +0.08). Plumbing: `e75_run.py` reproduced the exploration on s1
(gaps 0.0114, 0.0299, 0.0116; corr -0.37).
**The idea.** B's single pooling head makes its signed preference over the source's past errors depend weakly on the source's current
reading; the downstream relies on that reading-dependent part of the pooled statistic (a small change of attention, a large change of
KL when it is broken). A's opposite-sign pair forms the even statistic without the reading, so A nets barely need it for KL, although it
lowers their departure (E72). The consensus head's reading-dependence is not what the B nets need.
**Link to M2 (how transformers approximate Bayes; why the types differ).** This tests a type-specific computation inside the pooling
head: whether one attention head can tie its pooling to the token's own content in a way the rest of the net uses, and whether that is
what separates the B type's reading of its pool from the A type's (E72, E73).
**Design** (`e75_run.py`, `e75_summ.py`; `e63_run.py`, `pfn_b1hw.py`, `pfn_l2fix.py`, `e48_fmt.py`, `pfn2.py` unchanged). Nets with pfn2's
standard forward: registered cohorts ctrl (E37), fix (E30), h4 (E69), b1h4 (E70), mag (E37), w64ctrl, w128ctrl, w64mag, w128mag (E38);
descriptive d3ctrl, d3mag; secondary E71's h2w8, h3w16, h4w16. Acquired nets (registered gap < 0.10), types from the registered evaluations.
Per net, in its own format: base, shuf_pool (the reading replaced by the next episode's in the pooling heads' summary-row queries) and
shuf_cons (in the consensus heads'): test-NLL gap on 16,000 test episodes, rom on E12's pairs; corr(K, o*) per pooling head on E27's
first 2,000 bases. One analysis-queue job per cohort (nice 19).
**Predictions** (sets: A = A nets with >= 2 pooling heads; Bp = B nets with >= 1 pooling head; Bc = B nets with >= 1 consensus head).
- P302: in Bp, gap(shuf_pool) / gap(base) >= 1.5 in >= 75%.
- P303: in A, gap(shuf_pool) / gap(base) <= 1.15 in >= 75%.
- P304: in Bc, gap(shuf_cons) / gap(base) <= 1.10 in >= 75%.
- P305: in A, rom_large rises by >= 0.02 under shuf_pool in >= 75%.
- P306: corr(K, o*) <= -0.20 for >= 75% of Bp nets' pooling heads, and |corr(K, o*)| <= 0.15 for >= 75% of A nets' pooling heads.
**What the outcomes would mean.** P302-P304 hold: the B type's single pooling head carries reading-dependent information that its
downstream uses and the A type's paired heads do not, a functional difference between the types inside block 1's pooling, located in the
pooling head rather than the consensus head. P302 fails: B nets do not depend on it generally (the dev pattern was specific). P303 fails:
A nets need it too. P304 fails: the consensus head's query also carries it. P305 holds: the path lowers the A departure in the forward
computation as in the derivative (E72). P306 holds: the reading modulates the B pooling head's sign preference specifically.
**Limits.** Evaluation-time interventions off the trained operating point (the shuffled reading is real data but mismatched); the
attention changes are small, so what the downstream reads is not identified; standard-forward nets only (not own, nocons, uni).


## Amendment 84 (2026-10-31 22:21Z): E76, is the departure and its reading part fixed by block 1? A fresh reader trained on a frozen block 1, and the trained reader trained further (CPU, retraining)

Written by the eighth agent before any E76 training on a registered source. Known beforehand: E1-E75 with their reviews (E75: Update 85),
in particular E72's decomposition (rom = M3 + (detq - M3) + (rom - detq); the reading terms are similar across architectures), E73 (the
B nets' detq - M3 grows over training), E75 (B's reader depends on a reading-consistent pooling query), E61 and E75x (the best nets stay
near 0.003 nats from Bayes at widths 32-128 and depth 3, while the best 4-component output head is 0.00003 nats from the exact predictive,
`e75x_headfloor.py`: K = 1 0.0092, K = 2 0.00031, K = 4 0.000029 on 2,000 episodes; disclosed here).
**Exploration (disclosed; dev nets only, excluded).** `train_reader.py` on E43's newinit s0 (A) and s1 (B): with the embedding and block 1
frozen, a fresh reader (blocks 2.., the final LayerNorm and the head re-initialised, train_sw's schedule, a new episode stream) reached gap
0.0072 and 0.0109 (sources 0.0070, 0.0114), rom_large 0.086 and 0.122 (0.082, 0.133), detq 0.125 and 0.172 (0.123, 0.181), rom - detq
-0.039 and -0.050 (-0.041, -0.048); the trained reader trained further for 90,000 steps at lr 1e-4 reached gap 0.0059 and 0.0108 with
rom_large 0.081 and 0.122 (`results/e76x/*.json`, `results/e76x_eval.log`). Plumbing: 1,000-step smoke runs left the embedding and block 1
bit-identical and changed the reader.
**The idea.** The departure and both reading terms of E72's decomposition are a function of block 1's representation under this
recipe: a reader trained from scratch on the frozen block 1 converges to the same reading (A's excess over the benchmark, B's shortfall,
the query term), and the trained reader is close to converged in the departure, while more reader training still lowers the KL a little.
**Link to M2 (why they deviate; why they cannot learn better).** If a fresh reader reproduces the departure, the departure is set by the
first layer's computation, not by the joint training path of the reader, which locates the departure's cause in block 1 for both types;
whether more training of the reader lowers the KL tests whether the remaining distance is a training budget rather than a fixed limit.
**Design** (`train_reader.py`, `e76_eval.py`, `e76_summ.py`; `e72_detq.py`, `e63_run.py`, `e30_eval.py`, `e48_fmt.py`, `pfn2.py`,
`world.py` unchanged). Sources: E37's acquired ctrl nets with a pooling head, A s0, s1, s2, s6 and B s4, s5, s7, s9, s10, s11 (s8 has no
pooling head). Per source: fresh (`--mode fresh`, the reader re-initialised with torch.manual_seed(seed + 7600), 1,000-step warmup, lr 1e-3
to 80,000 steps, cooldown to 90,000) and continue (`--mode continue`, lr 1e-4 to 80,000, cooldown to 90,000); episodes from
np.random.default_rng([seed + 7600, cid, 7]). 20 trainings on General's CPU, half in the analysis queue now and half in the training queue
after E74's. Evaluation: `e76_eval.py` (gap on 16,000 test episodes, rom, detq) on the sources and readers; M3 of the source from E72.
**Predictions** (`e76_summ.py`).
- P307: fresh: |rom_large(fresh) - rom_large(source)| <= 0.02 in >= 75% of the 10 sources.
- P308: fresh: |detq(fresh) - detq(source)| <= 0.02 and |(rom - detq)(fresh) - (rom - detq)(source)| <= 0.015 in >= 75%.
- P309: fresh: gap(fresh) / gap(source) within [0.85, 1.15] in >= 75%.
- P310: continue: gap(continue) <= 0.95 x gap(source) and |rom_large(continue) - rom_large(source)| <= 0.02 in >= 75%.
**What the outcomes would mean.** P307-P309 hold: under this recipe the departure, its reading terms and the KL are fixed by the frozen
block 1 (a reader from scratch finds the same reading), for A and B alike; B's shortfall against the benchmark and A's excess are what a
trained reader makes of these block-1 representations. P307 or P308 fails: the reading depends on the joint training path (a fresh reader
reads the same pool differently). P309 fails: the KL depends on the reader's training path. P310 holds: more reader training lowers the
KL without moving the departure (part of the remaining distance is a training budget); fails: the trained reader is converged.
**Limits.** One recipe, width 32; a frozen block 1 cannot adapt to the new reader (so this does not test the joint optimum); 10 sources
sharing E37's seeds; the continued reader's schedule is one choice (lr 1e-4).
