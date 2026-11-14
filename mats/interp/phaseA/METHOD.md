# Replacing a layer by a simple distribution: method for the tiny PFN (2026-10-28)

Program agreed with Kelvin on 2026-10-28. This document merges three independent designs:
- **E**, estimation theory (use density ratios only where nothing else works; bridge estimator);
- **M**, mechanistic (q as a short program of world latents; telescoping consistency; subset swaps);
- **D**, description length (frontier, stopping rules, limits; relation to causal scrubbing and ARC surprise accounting).

Section 1 lists the points where they disagree and the decision taken on each. Sections 2–10 give the method. Section 11 is the round-1 plan. Nothing has been run except the two pre-method checks in §2.4.

---

## 1. Decisions where the designs disagree

| # | Issue | E | M | D | Decision and reason |
|---|---|---|---|---|---|
| 1 | Main closure number | Paired direct Δ, plus E_c[Δ(c)²] from two independent inner halves | Direct Δ | Paired faithfulness D_f = E KL(f(h) ‖ f(h')) ≥ 0 | **Primary: signed Δ and the per-cell RMS gap Δ_rms = sqrt(E_c[Δ(c)²]).** Δ_rms cannot cancel across c, and q = p(h\|c) gives Δ_rms = 0, so the zero point is right. D_f is never zero unless q copies the network episode by episode: the exact conditional sampler already has D_f > 0 from nuisance variation. So D_f is **secondary**: it is reported together with its floor (§4.4) and used only for rungs that claim per-episode nuisance effects. |
| 2 | What q may read | z must be a function of x, must contain the Bayes statistic b, and must leave enough randomness for a density | q(h\|z) = E_{u\|z} q(h\|u,z): read nuisances u but marginalise them, with z = (Q,S) | q(h\|x) conditioned on library features of the whole episode | **q may read any declared function φ(x) ⊇ c = (Q,S) of the inputs, including nuisances (IDs, residual directions), but never the true v (leakage; R1 in §2.3).** The conditioning set of the classifier and of the tempering path is the smaller set ζ (c at full cuts), with q marginalised over the rest of φ. Reading nuisances is a modelling choice. Conditioning the ratio on them is not allowed: that makes p(h\|ζ) singular. |
| 3 | Density ratio on 512-d layers | Undefined when p(h\|z) is singular, so restrict or smooth | Use classifiers anyway; ESS gate | Always smooth at a declared σ | **Two ratios.** (a) The *output-space* ratio w_out = p(o\|c)/q(o\|c), with o the 11 free head parameters. G depends on h only through o, so this ratio is G-sufficient, exists at every full cut, and has χ²_out ≤ χ²_h (data processing). It is used for the tempering path, the identity checks and G-relevant localisation. (b) The *h-space* ratio is used only to name structure (which direction or feature). It is used only where T8 shows a density (single-token marginals; QT sites), or else at the declared smoothing σ (§3.6). |
| 4 | Tempering estimator | Bridge (pool p and q; weights c^α(1−c)^(1−α), bounded); normalised globally | SNIS from q with M draws per z, normalised per z | SNIS from q | **Bridge weights, normalised per c.** Per-c normalisation keeps p(c) fixed, so the label/target distribution is unchanged along the path. Global normalisation reweights episodes by Z(α,c) and mixes "different structure" with "different episodes". The exact conditional sampler gives M real draws at each c, so the bridge can be run inside each c. |
| 5 | Sites | QT after the last block (32); QT after block 2 of the 3-layer model; 3 QQ tokens after block 1 (96), rest real | Full layers 0, 1a, 1, 2a, 2 | Full cut after block 1, modelling the block's write | **Use all of them, by role** (§2.2). Low-dim full cuts (QT after attn2 / block 2) validate the ratio machinery. The full cut after block 1 is the main ladder site, scored by direct estimates. The QQ partial site carries the interface split. Partial sites always report the interface term. |
| 6 | q0 content | Linear in own-source statistics plus IDs linearly ("layer 2 needs them for matching") | R0 constant, R1 raw inputs, R2 Bayes quantities | q_lin-in, then q0 = + Bayes latents | **q0 = Gaussian, mean linear in [token's raw inputs incl. IDs] + [Bayes latents of the token's source / episode].** Without the ID terms, layer 2 cannot route to the synthetic tokens, and Δ just re-measures binding (established in the binding paper). "No-ID q0" is run once as a control. R_marg and R_in are anchors below q0. |
| 7 | Description length | (#params/2)·log2 n | Program bits in a grammar + parameters quantised to the coarsest precision that moves Δ by < 1 SE | Grammar bits + pruned, quantised parameters; BIC cross-check; claim an ordering only if both codes agree | **D's scheme, with M's quantisation tolerance** (§8). Library features cost selection bits; their compute is reported as sampling cost. The exchange-rate (two-part) view is reported as a break-even N\*, not used for stopping: at N = 16k one 8-bit parameter costs 3.5e-4 nats per episode, so a two-part code would reject every rung of a description of a 0.01-nat error. |
| 8 | Cross-layer test | Classifier F#q_l vs q_{l+1}, AUC ≤ 0.55 | Exact telescoping Δ_l = Δ_{l+1} + Γ; derived chain with short surrogate blocks F̃ | Bits: DL(q_l) + DL(M_l) + DL(Corr) | **Three tiers** (§6): (i) the identity Δ_l(q_l) = Δ_{l+1}(F#q_l) (a plumbing check); (ii) actual-computation consistency: q̃_{l+1} = Proj(F#q_l) refitted in the same family must match the real-fitted q_{l+1}; (iii) mechanistic compression via F̃ with DL_chain. Tier (ii) is Kelvin's condition. Tier (iii) is what turns it into understanding. |
| 9 | Smoothing σ rule | Smoothing bias < 0.1·Δ | – | Largest σ with L_σ(p) − L(p) ≤ 5% of KL_model | **D's rule.** It fixes σ once per site before the ladder; E's rule would move σ with every rung. |
| 10 | Stopping | TOST on Δ and Δ_rms | Δ ≤ 0.1·KL_model, or 3 rejected rounds | S1 closed / S2 flat / S3 no compression | **S1–S3 of D, with E's TOST as the S1 test and M's acceptance rule in the loop** (§9). The S3 budget is DL(q) ≤ 0.5·DL_net(block) rather than 0.1: block 1 has only ~8.5k parameters, and 0.1 would trigger on bookkeeping alone. |
| 11 | Naming target | Regress held-out log ŵ on candidates; rank by G(w−1) | Lasso/tree on log ŵ and per-episode contributions over a fixed library | Classify the residual as feature- vs shape-insufficient with a resampling oracle | **The target is the G-relevant gap:** cell gaps Δ(c) and output-space log ŵ_out. h-space log ŵ only says *where* in h. Acceptance always uses fresh data and the direct Δ_rms. D's oracle 2×2 decides whether to add features or add shape. |

---

## 2. Objects

### 2.1 World and statistics
- An episode is x = (v, y, o, IDs).
- c(x) = (Q, S) ∈ R^6, with Q_s = o*_s (the query readings) and S_s = Σ_{j≤4} (o_sj − y_j)² (labelled residual sums of squares). c is sufficient: p(y*\|x) = p_B(y*\|c).
- The nuisances u given c have 141 dimensions:
  - source IDs (3×16) and question IDs (5×16);
  - residual directions r̂_s ∈ S³ (3×3);
  - labelled y (4).
  - The network input has 147 = 141 + 6 dimensions.
- **Exact conditional sampler.** For any ψ ⊇ c built from world variables, draw x' ~ p(x\|ψ(x)):
  - keep ψ fixed;
  - r_s = sqrt(S_s)·Unif(S³);
  - y_lab ~ N(0, I₄), and o_lab = y_lab + r;
  - IDs not in ψ ~ N(0, I/16);
  - o* = Q.

  This is exact because r̂_s, y_lab and the IDs are independent of v and Q given S. B2 already uses it (`budget.py:draw`). Unit tests: Bayes invariance (B2 max change 8e-15), and a KS test of r̂ against uniform.
- Token layout: t = 5s + j. LAB tokens are j < 4 (12 tokens), QQ_s is t = 5s + 4, QT is t = 15.

### 2.2 Sites
A site is (layer cut, token group g). h = h_g(x), and the other tokens stay real.

A **full cut** contains every token still read downstream. At a full cut the interface term (§4.3) is identically 0.

2-layer w32_h2_L2 (mixture head):

| Site | h | Read downstream by | Effective dim | Role |
|---|---|---|---|---|
| S0 | embedding, 16×32 | everything | – | Plumbing check: R_in is exact |
| S1 | all tokens after block 1 (full cut) | only QT's layer-2 attention reads context tokens (k, v through LN); QT also through its residual stream | 15×30 + 31 = 481 | Main ladder site; direct estimates only for h-space |
| S1-QQ | 3 QQ tokens after block 1, rest real | as S1 | 3×30 | Reliability code; interface split |
| S2a | QT after attn2 (full cut: context tokens are dead) | mlp2, head | 31 | Low-dim; validates ratio machinery |
| S2 | QT after block 2 (full cut) | LN + linear head (11 free outputs) | ≤ 31 (G sees 11 + norm) | Link to the B2 output-moment split |

3-layer L3_ws (round 2, except E5b):
- T1: after block 1, all tokens. Only the shift quotient applies, because block 2 updates context tokens.
- T2: after block 2, all tokens. Same quotient as S1.
- T2-QT: partial site (E's "site B").
- T3a: QT after attn3 (full cut).
- T3: QT after block 3 (full cut).

### 2.3 Downstream quantity and the q family
- **G(h, x) = KL(p_B(·\|c) ‖ f_{>l}(h, r(x)))**, by quadrature.
  - It is Rao-Blackwellised over y*: per-episode sd ≈ 0.03 nats, against 0.16 for NLL at a sampled y*.
  - Its expectation under real h at a full cut is the model's KL (0.0071–0.0144 over the main seeds).
  - Also reported: the vector G⁺ = (G, predictive mean, predictive log sd) for moment-wise gaps.
- **q_k(h \| φ_k(x))**, where φ_k ⊇ c is a declared feature map.
  - R1, no leakage: φ_k is a function of x. The true v is never an input.
  - R2, valid labels: c ⊆ φ_k, so the Rao-Blackwellised target p_B(·\|c) stays valid for synthetic h.
  - Equivalently, q_k(h\|c) = E_{u\|c} q_k(h\|φ_k(c,u)). To sample: draw u from the exact world conditional, compute φ_k, then sample q_k.
- **R3, density existence (for h-space ratios only).** p(h\|ζ) has a density only if the randomness left given ζ is at least the effective dimension.
  - ζ = c leaves 141 dims. That covers S2/S2a (31), single tokens (30) and S1-QQ (90).
  - It does not cover S1 (481), so S1 has no density.
  - ζ = c + IDs leaves 13 dims, so no multi-token site has a density.
  - T8 checks this by intrinsic dimension (§5).

### 2.4 Checked before writing (`checks/quotient_timing.py`, main s0, 4,000 test episodes)
- **Quotient.**
  - Adding c·1 to any token at S1 changes the head outputs by at most 1.2e-6 (exactly invariant up to float error).
  - Rescaling centred *context* tokens by a ∈ [0.5, 2.5] changes them by at most 1e-4: invariant up to LayerNorm's eps.
  - Rescaling QT (control) changes the logits by 0.9.
  - So q is fitted on per-token LN-normalised states ĥ_t = (h_t − mean)/std for context tokens at S1 and T2, and on centred h for QT. The dimensions in the table above already subtract these quotients.
- **Cost.**
  - Forward from S1 to the head: 3.1e-5 s/episode.
  - The existing `kl_grid` (2001 points over Bayes and model support): 8.4e-3 s/episode.
  - The integrand of KL(p_B ‖ f) is weighted by p_B, so a grid over Bayes ±9 sd suffices. At 201 points it costs 8.7e-4 s/episode, and it differs from `kl_grid` by at most 1.5e-16 on real outputs.
  - The Bayes log-density on the grid depends only on c, so it is computed once per c and shared by all M inner draws.
  - Re-verify on 2,000 synthetic samples per rung, because off-manifold outputs may be odd. Pass: max\|diff\| < 1e-6.

---

## 3. Estimators

Notation: Δ(q) = L(q) − L(p), with L(q) = E_x E_{h'~q(·\|φ(x))} G(h', x) and L(p) = E_x G(h(x), x). This is Kelvin's L(0) − L(1), and it equals −Cov_q(G, w). Δ > 0 means q loses more than the network. **Δ < 0 is possible**, for example when q lacks the network's ID crosstalk. It is a finding, not closure.

### 3.1 Direct gap (no weights)
- Use episodes i = 1..n from a fresh split, with M synthetic draws each (common random numbers across rungs: the same x_i, u-draws and noise seeds).
- d_i = (1/M) Σ_m G(h'_im, x_i) − G(h(x_i), x_i), and Δ̂ = mean d_i.
- The same formula with G⁺ gives the mean / log-sd gaps.
- **CI:** paired studentised bootstrap over episodes (B = 2000).
- **Tail check:** Hill estimate of the tail index of \|d_i\| (top 2%). If it is below 3, double n and report it. The top 10% of episodes carry 43% of the model's KL, so the tails are heavy.

### 3.2 Per-cell gap and Δ_rms (no cancellation)
- For each evaluation c_i:
  - draw M_p real states from the exact conditional sampler x'_ij ~ p(x\|c_i) and run the network (at partial sites only h_g comes from x', the rest from x_i: this is p̃, §4.3);
  - draw M_q states from q(·\|c_i).
- Split the draws into two independent halves A and B. With Δ̂^A(c_i) = mean_A G_q − mean_A G_p (and the same for B), **E_c[Δ(c)²] is estimated without bias by mean_i Δ̂^A(c_i)·Δ̂^B(c_i)**.
- Δ_rms = sqrt(max(0, ·)). Report the CI by bootstrap over i.
- At the level of a finer world-variable set ψ (e.g. c + IDs), the same estimator with the sampler conditioned on ψ gives Δ_rms^ψ. Use it for rungs that claim nuisance-dependent structure.
- Default sizes: n = 50k cells and M_p = M_q = 8 (4 + 4). R1.0 fixes the final sizes (§11).

### 3.3 Ratio estimation (classifiers)
- **Data.** Pairs (real, synthetic) share the same ζ, with balanced classes. Use three disjoint fresh splits per seed:
  - fit q: 50k episodes;
  - train the classifier: 100k per class;
  - estimate: 50k cells with inner draws.

  Episode seeds start at 60000, disjoint from val 10000, test 20000, binding 40000 and the training streams.
- **Families** (two are mandatory; the third is a cross-check):
  1. Quadratic logistic in whitened u = Σ^{-1/2}(h − μ(φ)), with c-interactions. It is exact if p(h\|ζ) is Gaussian, and it is the closed-form QDA baseline.
  2. An equivariant MLP.
     - QT and output sites: an MLP on (h or o, features of c).
     - Multi-token sites: nested DeepSets: per-token MLP on [ĥ_t, type, o*_{s(t)}, log S_{s(t)}] → pool over a source's LAB tokens → combine with QQ_s → pool over sources → add QT and c → MLP.
     - Temperature calibration on held-out data.
  3. uLSIF with Gaussian kernels, only at d ≤ 31.
- **Outputs.**
  - Held-out AUC and BCE. JS ≥ log 2 − BCE.
  - Donsker–Varadhan bound KL ≥ E_p T − log E_q e^T.
  - Plug-in KL̂ = E_p log ŵ.
  - χ̂² only when the PSIS k̂ < 0.5.
  - Per-c ESS.
- **Output-space classifier** on (o, c), with o the 11 free head parameters (4 means, 4 log sds, 4 logits minus one shift). Density exists at every full cut, and its log ŵ_out is the G-sufficient ratio: E_q[G w_out] = E_p[G] exactly, because G is a function of (o, c).
- **Density chasm** (held-out AUC > 0.99): telescope. log w = Σ_k log(q_{k+1}/q_k), which is exact because the rungs have known densities when they do not read nuisances, plus log(p/q_K) from the classifier. Otherwise report "separable at this n" and use only the direct estimates.

### 3.4 Tempering path L(α) (bridge, normalised per c)
- For each evaluation c_i, pool the M real draws (from p, or p̃ at partial sites) and the M synthetic draws. The pool is a sample from m_c = (p + q)/2.
- With the classifier ĉ = p/(p+q):
  - ω_j(α) = ĉ_j^α (1−ĉ_j)^{1−α}, with p_α/m = 2ω/Z(α,c) ≤ 2/Z(α,c);
  - L̂(α\|c_i) = Σ_j ω_j G_j / Σ_j ω_j, and Ẑ(α\|c_i) = (1/M) Σ_j ω_j;
  - L̂(α) = mean_i L̂(α\|c_i).
- Derivative: dL/dα = E_c Cov_{p_α(·\|c)}(G, log ŵ), and L(1) − L(0) = ∫₀¹ dL/dα dα. The integral must match the direct −Δ (check V5).
- Grid α ∈ {0, 0.1, …, 1}. Run it in output space (the default, always defined) and in h space where R3 holds.
- **Variance.**
  - Weights are bounded for every α, so variance is governed by the Rényi affinity Z(α) = exp(−(1−α) D_α(p‖q)), not by χ².
  - Delta-method variance per c: Σ ω²(G − L̂)² / (Σ ω)².
  - Report Kish ESS. Pooled over c, it must be ≥ 1000 (T4).
  - SNIS bias is O(1/M): compare M ∈ {8, 16, 32} on a 5k subset. The bias must be below 1 SE.
- **CI.**
  - Bootstrap over c with the classifier refitted in each replicate (B = 200 for the quadratic family, 50 for the MLP).
  - Add a systematic band = max(endpoint-closure error, spread across classifier families at α = ½).
- Kelvin's q-side identity E_q[G ŵ] = E_p[G] (plain IS from q) is computed only as diagnostic T2. Planning numbers in whitened coordinates for a 20% scale misfit (κ = 1.2):
  - E_q w² = (κ/sqrt(2κ−1))^d, which is 1.56 at d = 31, 3.7 at d = 93 and about 9e2 at d = 481;
  - Bhattacharyya Z(½) = 0.35 even at d = 512.

### 3.5 Localisation regions (direct; selection on one split, estimation on another)
- For any partition {A_k} of synthetic and real draws, Δ = Σ_k Δ_A with **Δ_A = E_q[G·1_A] − E_p[G·1_A]**. This is exact and needs no weights.
- Split: Δ_A = (P_q(A) − P_p(A))·E_p[G\|A] + P_q(A)·(E_q[G\|A] − E_p[G\|A]), i.e. a mass part plus a within-region part.
- Partitions:
  - **episode-space quintiles**, pre-registered: between-component variance fraction, query spread, coupling KL(w_B‖w_lab), number of reliable-looking sources;
  - **output-space ŵ_out deciles** and **h-space ŵ deciles**, chosen on split 2 and estimated on split 3;
  - **an off-support region** {ŵ < 0.05} on the q side. Its share of Δ measures how downstream extrapolates off the manifold, not missing structure.

### 3.6 Smoothing (only for h-space ratios where R3 fails)
- p_σ = p * N(0, σ²I), and the same noise is applied to synthetic draws, in quotient coordinates.
- σ is the largest value with L_σ(p) − L(p) ≤ 0.05·KL_model. It is fixed per site and seed before the ladder.
- All smoothed results are labelled "at resolution σ".

---

## 4. Decompositions used throughout

### 4.1 Signed and cell gaps
Δ, Δ_rms, Δ_rms^ψ and the moment gaps, as in §3.1–3.2.

### 4.2 Fragility control (off-manifold)
- q_noise(h\|x) = h(x) + ε, with ε ~ N(0, Σ_q,resid), the residual covariance of the rung being tested.
- If L(q_noise) − L(p) ≥ 0.5·Δ(q), most of Δ is how the network extrapolates under noise of that shape, not missing structure. The Gaussian base is then the wrong family for that token type, and the next rung must be on-manifold (a resample base, §7.3, or a smaller heteroscedastic noise).

### 4.3 Interface split (partial sites)
- p̃_φ replaces h_g by h_g(x') with x' ~ p(x\|φ(x)), and keeps the rest from x.
- Δ = [L(q) − L(p̃_φ)] + [L(p̃_φ) − L(p)], i.e. shape plus interface.
- The interface term is 0 at full cuts with φ ⊇ c, because resampling preserves the joint (h, c).
- At S1-QQ it measures whether φ carries everything the real rest needs to coordinate with g: ID matching first, then cross-token structure.
- The ladder for φ_iface is c → c + qid* → c + qid* + sids. The first φ with interface term ≤ 2 SE fixes ζ for the classifiers at that site.

### 4.4 Faithfulness (secondary)
- D_f(q) = E_x E_{h'~q} KL(f(h(x)) ‖ f(h')), paired on x.
- Its floor is D_f(p̃_φ), the per-episode spread of the exact conditional sampler at the same φ.
- Report D_f/KL_model and the excess D_f(q) − D_f(p̃_φ).
- It is used only to test rungs that claim per-episode nuisance structure (IDs, residual direction).

---

## 5. Trust checks (each reported number carries pass/fail)

| ID | Check | Pass |
|---|---|---|
| T1 | Bridge endpoints vs direct L(0), L(1) | within max(2 SE, 0.05·\|Δ\|) |
| T2 | Normalisation: E_q[ŵ] and E_p[1/ŵ] | each 1 ± 0.05 |
| T3 | PSIS k̂, both sides | < 0.7 for any q-side number; < 0.5 to quote χ² |
| T4 | Bridge Kish ESS (pooled over c) | ≥ 1000 |
| T5 | Calibration | ECE < 0.02; train − held-out BCE < 0.01 |
| T6 | AUC regime | > 0.99: telescope. Within 0.5 ± 0.01 with DV ≈ 0: "indistinguishable at n", not "q = p" |
| T7 | Classifier families agree on L(½) | within the bootstrap CI |
| T8 | Intrinsic dimension (TwoNN and local PCA) of exact conditional samples given ζ | ≥ effective dim − 2, else no h-space density: smooth or restrict |
| T9 | Off-support share | reported separately (§3.5) |
| T10 | Leakage | no v in φ; L(q) ≥ the L1 floor for that φ (§10) |
| V1 | S0 with R_in | Δ within 2 SE of 0, AUC ≤ 0.51 |
| V3 | Unseen-function identity: E_q[g ŵ_out] = E_p[g] for g ∈ {predictive mean, log sd}; E_q[ŵ \| c-bin] = 1 over 20 bins | within 2 SE |
| V4 | log ŵ stable across folds and classifier seeds | r > 0.9 |
| V5 | ∫ dL/dα = −Δ (direct) | within 2 SE |
| V6 | Exact conditional sampler at a full cut | Δ = 0 within 2 SE; AUC 0.5 ± 0.01 |
| V7 | Shuffled-latent q0 (c permuted across episodes) does not beat R_in | – |
| V8 | Untrained network: closed by R_in at small DL | S1 met |

---

## 6. Cross-layer consistency

F_{l+1} is the real block. Pushing a description forward, F#q_l, means running q_l's samples through F_{l+1}.

1. **Identity (tier i).** E_{F#q_l} G = L_l(q_l), so Δ_l(q_l) = Δ_{l+1}(F#q_l) exactly. It is checked numerically to float error and has no content beyond plumbing. Starting from q_0 = the exact embedding, F#q_0 = p_1, so actual-F consistency only has content after each layer's description is *projected* onto a short family.
2. **Actual-computation consistency (tier ii, Kelvin's condition).**
   - q̃_{l+1} = Proj(F#q_l): refit q_{l+1}'s family (same features, same noise class) on pushed samples only, never on real h_{l+1}.
   - Γ_proj = Δ_{l+1}(q̃_{l+1}) − Δ_{l+1}(F#q_l): what the family loses when summarising the pushforward.
   - Γ_fit = Δ_rms(q̃_{l+1}) − Δ_rms(q^fit_{l+1}): structure that the real-fitted description has and the derived one lacks.
   - AUC(q̃_{l+1} vs q^fit_{l+1}). Both samplers are synthetic, so n is unlimited and both have densities.
   - **Pass: Γ_fit ≤ max(2 SE, 0.1·KL_model) and AUC ≤ 0.55.**
   - On fail, the extra structure in q^fit_{l+1} is "representation not derived from layer-l's description". It becomes a hypothesis about block l+1 or a missing feature at layer l.
   - Only meaningful when q_l itself is near-closed. A bad q_l fails trivially.
3. **Mechanistic compression (tier iii).**
   - Replace F_{l+1} by a short F̃, and set q̃_{l+1} = Proj(F̃#q_l). Candidates from prior work:
     - F̃_1: within-source QQ attention softmax(c₁r + c₂r²) (two scalars per head; exact to 0.001 nats in 6 of 7 nets), plus the query-mixer head in s0/s2/s3, then MLP1 ≈ g(features) with a small g;
     - F̃_2: QT softmax over QQ with logits a·log β̂_s + b, values o*_s;
     - F̃_3: the same with a ≈ 0.8.
   - DL_chain = DL(q_0) + Σ DL(F̃_l), and Δ_chain is measured at the last cut.
   - Pass per layer: Γ̃ ≤ 0.1·KL_model. Report each chain on the (DL, Δ_rms) frontier.

---

## 7. q-ladder for the tiny PFN

### 7.1 Shared form
- Per token t of type τ ∈ {LAB, QQ, QT}, in quotient coordinates: **h_t = A_τ φ_τ(t; x) + b_τ + ε_t.**
- A_τ is shared across sources and questions. Equivariance is verified (max deviation 1.5e-5).
- Noise: ε_t = C_τ ξ_{s(t)} + D_τ ζ_{j(t)} + E_τ η + diag-Gaussian. The terms are a per-source factor ξ, a per-question factor ζ and an episode factor η, each of rank ≤ 4.
  - Rungs below q1f use the diagonal term plus a rank-≤4 within-token factor only, with tokens independent given φ.
- Fit by ridge regression (λ by 5-fold CV grouped by episode) on the fit split. Residual covariance by PPCA/factor analysis in closed form, and EM for mixtures.

### 7.2 Feature library Λ (fixed in PREREG before any ladder result)
- **Token raw:** sid_{s(t)}, qid_{j(t)}, o_t, y_t, flags; r_t = o_t − y_t at LAB.
- **Own-source Bayes:** log S_s, E_lab,s, p_lab,s(v) ∈ R³, o*_s, o*_s·E_lab,s.
- **Cross-source:** o*_{−s} (the two other readings), β_s, p_B,s(v), dE_s = E_B,s − E_lab,s.
- **Episode / QT:** mean(Q), Bayes mean, Bayes log variance, lab-only mean, between-component variance fraction, 27-way entropy, query spread, coupling KL(w_B‖w_lab).
- **Mechanism:**
  - tilted pools m_{s±} = Σ_j softmax_j(±c₁r_sj + c₂r_sj²)·[r_sj, r_sj²], with (c₁, c₂) from step 3 per head;
  - flattened weights β^a;
  - token r², and mean r³ per source.
- **Nuisance:** ID Grams ⟨sid_s, sid_s'⟩ and ⟨sid_s, qid_j⟩; residual-direction statistics.

### 7.3 Rungs (per site; S2/S2a use only the QT rows)

| Rung | Adds | Rationale / prior evidence |
|---|---|---|
| R_marg | per-type constant mean | anchor; L1-floor check |
| R_in | linear in token raw inputs | exact at S0; "the block did nothing nonlinear"; untrained control (V8) |
| **q0** | R_in + own-source Bayes latents at LAB/QQ; Bayes predictive moments at QT | Kelvin's q0; IDs are kept for routing (§1 #6) |
| q0−ID | q0 without ID terms (control, once) | measures the routing failure |
| q1a | o*_{−s}, β_s, p_B,s at QQ | in one-pool seeds MLP1 at QQ needs the other readings (B2: KL 0.19 → within 0.004–0.009) |
| q1b | tilted pools m_{s±} | residual-direction share 21% (9% two-pool, 25–32% one-pool) |
| q1c | ID Grams | ID crosstalk 22% of KL |
| q1d | token r, r² at LAB | r² decodable after mlp1 (R² 0.89) |
| q1e | heteroscedastic Σ(ψ); mixture over per-source reliability hypotheses gated by p_B(v_s\|c) | systematic KL concentrated in multimodal episodes (top between-component-variance quintile holds 48%) |
| q1f | shared per-source / per-question / episode factors | cross-token covariance |
| q^R (oracle, not a frontier point) | each token type (or each source's tokens) resampled independently from real episodes matching its features in bins | tests the factorisation and feature claims (§10, 2×2) |
| R4+ | named features from the large-w loop (§7.4) | – |

- Report every rung **cumulatively** (q0 → q0+a → q0+a+b …, in the order above) for the frontier, and **add-one to q0** for attribution.
- With ≤ 4 groups, also report Shapley values over {a, b, c, e}.

### 7.4 Large-w → named hypothesis (one round = one leaf; at most 3 rounds per site)
1. Locate:
   - output-space ŵ_out regions, and which moment they move (mean / variance / shape, matching B2's split);
   - token type and source from the DeepSets classifier's additive structure;
   - the relative-Fisher subspace U_τ = top eigenvectors of E_p[∇_{ĥ_τ} log ŵ ∇ log ŵᵀ].
2. Regress the cell gaps Δ̂(c) (two-half product weights), log ŵ_out and Uᵀ(ĥ − μ) on Λ, with lasso (≤ 5 terms) and a depth-3 tree. Fit on split 2 and score on split 3.
3. State the hypothesis H_k as "ψ enters h_τ along U_τ". A discrete or multimodal ψ enters as a mixture; a ψ that changes the spread enters as Σ(ψ).
4. **Accept on fresh data** if all of these hold:
   - Δ_rms falls by ≥ max(10% of current, 3 SE);
   - the refitted classifier's lasso no longer selects ψ;
   - Δ_rms reduction per added bit beats the best rival ψ.

   Otherwise reject. Three rejected rounds count as S2.

---

## 8. Description length and sampling cost

- **DL(q) = DL_struct + DL_param**, with C(q) (FLOPs per episode, including feature computation) reported beside it.
- **DL_struct**, per token type, in a fixed grammar declared in PREREG:
  - features: k selected from Λ at log2\|Λ\| bits each; new loop features at (#primitive tokens × log2\|primitives\|), from primitives {r, r², Σ_own-source, Σ_sources, log, exp, softmax_sources, o*, Bayes posterior ops};
  - map type: linear / spline-4 / MLP-8 (log2 3);
  - noise type: iso / diag / rank-r / full × homo / hetero (log2 8, plus log2 r_max);
  - mixture K and gate features;
  - factor structure (log2 4);
  - hand-set scalars at 8 bits each.
- **DL_param:**
  - prune each tensor to the smallest rank that moves L(q) and Δ_rms by ≤ max(1 SE, 0.02·KL_model);
  - quantise uniformly at b bits with an fp16 scale per tensor; b* is the smallest b within the same tolerance;
  - DL_param = Σ n_eff·b* + 16·n_tensors.
  - Cross-check with BIC, (k/2)·log2 N_fit. **An ordering of two rungs is claimed only if both codes agree.**
- **Free givens:** the world sampler, the exact conditional sampler, the embedding h_0 (shared by every description), and the definitions in Λ. Each *use* of a library feature costs selection bits. Its compute goes into C(q).
  - A rung that copies Bayes quantities says "layer l holds Bayes features". It says *how* only once tier-iii consistency derives them.
- **Network anchor:** DL_net(block l) uses the same quantisation rule on the block's weights (block 1 ≈ 8.5k params; up to layer 1 ≈ 9.7k with W_in). Compression κ = DL_net/DL(q).
- **Structural vs parametric bits** are reported separately. Structural bits are what should transfer across seeds; parametric bits mostly locate features in a seed's basis.
  - Transfer test V9: refit only parameters on seeds 5, 7 and 9. Pass: Δ_rms ≤ 1.5× its value on seeds 0–4.

---

## 9. Frontier and stopping (per site, per seed)

- **Frontier:** the lower convex hull of (DL, Δ_rms) over ladder rungs and accepted loop rungs, with signed Δ, D_f and C(q) annotated.
- **Break-even N\*** = ΔDL·ln 2 / (reduction in Δ_rms): the number of episodes after which a rung pays for its bits in a two-part code. This is the ARC surprise-accounting view.
- **Stop when one of these holds:**
  - **S1, closed.** TOST: the 90% CI of Δ ⊂ [−ε, ε] **and** the upper 95% bound of Δ_rms ≤ ε, with ε = 0.1·KL_model(seed), about 0.001 nats. Also, the signed Δ is within 2 SE in every pre-registered quintile.
  - **S2, flat.** Two consecutive pre-registered rungs, and the naming loop, each fail the acceptance rule of §7.4.
  - **S3, budget.** The next rung would exceed DL = 0.5·DL_net(block) while S1 is unmet.
- **An "understood" claim at layer l requires all of:**
  - S1 at DL ≤ 0.5·DL_net(block);
  - C(q) ≤ C(network up to l);
  - a tier-ii consistency pass with the neighbouring layers.

---

## 10. What counts as a fundamental limit

Every residual is classified with its check:

- **L1, information floor (exact).** If q's output depends on x only through φ, then L(q) ≥ I(Y*; c \| φ).
  - Exact values: φ = Q gives 0.185; (Q, U) with U the unordered S gives 0.126; (Q, ΣS) gives 0.155; c gives 0.
  - These double as pipeline checks (T10). L1 is a real limit only when a rung must drop information.
- **L2, factorisation/feature floor (2×2 with the oracle q^R).**
  - At a full cut, p(h\|c) closes trivially. Causal-scrubbing content is only in factorisation claims such as "tokens independent given features".
  - Oracle closes, parametric q does not: **shape-insufficient** (add mixture or heteroscedastic shape).
  - The oracle itself fails: **feature/factorisation-insufficient** (add features or shared factors).
  - Bin-width sensitivity is reported at two widths.
- **L3, budget floor.** S2 or S3 is reached. This is an upper bound on min Δ_rms over q with DL ≤ B. It is relative to the grammar and the ladder, never a theorem.
  - It is called **"no short description found"** when it persists through the naming loop and the per-c residual Δ(c) is shared across seeds (r ≥ 0.6). In that case the structure is real but has no short description in this grammar.
- **L4, estimator floor.**
  - \|Δ\| < 2 SE at the largest affordable n (about 2e-4 nats).
  - h-space ratios undefined (T8 fails) or untrusted (T1–T7 fail).
  - AUC ≈ 0.5 while Δ_rms > ε: the structure that matters is invisible to the classifier, so use the output-space ratio. This limits claims, not the network.
- **L5, consistency failure.** q closes at l and at l+1, but the derived q̃_{l+1} does not: representation, not computation.
- **Nuisance note.** About 43% of KL_model comes from IDs and residual direction. A q that does not read nuisances can match it only in distribution (Δ_rms at level c). Per-episode matching (D_f, Δ_rms^ψ) needs nuisance-reading rungs.

---

## 11. Round-1 plan (CPU only, ≤ 4 processes, OMP_NUM_THREADS=1)

Checkpoints:
- **Main 2-layer:** `tiny-pfn/runs/w32_h2_L2_mix4_ws_s{0..4}`. Motifs: s1, s4 have two pooling heads; s0, s2, s3 have one pooling head plus a mixer.
- **Controls:**
  - plateau `tiny-pfn/runs/w32_h2_L2_mix4_s3`;
  - third solution `tiny-pfn-interp/b1/runs/w32_h2_L2_mix4_ws_s8`;
  - untrained nets built under the same seeds.
- **Transfer:** `b1/runs/w32_h2_L2_mix4_ws_s{5,7,9}`.
- **3-layer:** `tiny-pfn-interp/heads/hruns/L3_ws_s10..14`, and `base_ws_s{10,12,13,14}` (s11 is plateau, excluded).

Code goes in `cm-interp/lib/`: sites, sampler, gquad, qfam, clf, bridge.

Rough cost: the G quadrature dominates. At about 3e-4 s per draw with the Bayes grid shared within c, one (seed, site, rung) cell with n = 50k and 8 + 8 inner draws takes about 5–8 min on one process.

| Step | What | Checkpoints | Numbers that decide |
|---|---|---|---|
| **R1.0** (gate) | Plumbing and calibration. (a) G quadrature vs `eval.json` KL. (b) Quotient and grid checks (§2.4) on synthetic draws. (c) Exact-sampler unit tests. (d) V1 at S0, V6 at S1 and S2a. (e) X0 known truth: analytic Gaussians at d ∈ {31, 90}, planted ψ = S_s·v added to q0 at S1-QQ with γ set so that Δ ∈ {0.001, 0.01}, and the null p = q, 50 replicates. (f) Variance pilot: per-episode sd of d_i and of the half-products. Then write **PREREG.md** (ε, Λ, grammar, sizes, predictions P1–P6, episode-seed ranges) before any ladder result. | main s0 (all); s0–s4 for (a) | Proceed only if: V1 and V6 pass; X0 coverage ≥ 0.9 wherever T1–T9 pass; T flags fire wherever bias exceeds the CI; planted ψ recovered (cos(v̂, v) > 0.9, ψ ranked first by the loop). Sample sizes for se(Δ) ≤ 3e-4 and the TOST at ε are fixed here. |
| **R1.1** | Low-dim full cuts S2a and S2: rungs R_marg, R_in, q0, q1e. Direct Δ, Δ_rms, moment gaps; output- and h-space classifiers (all three families); bridge L(α) with derivative; T1–T7; regions. | main s0–s4 | **Machinery decision:** if T1, T2, T5 and T7 pass for q0 at S2a in ≥ 4/5 seeds, ratio-based path and localisation are trusted at d ≤ 31; otherwise ratio results are diagnostics only and the program runs on direct estimates. **P1:** q0 does not reach S1 at S2 (Δ_rms ≥ 0.2·KL_model in ≥ 4/5 seeds). If it does, the readout state is a linear-Gaussian function of Bayes moments, which conflicts with the 0.3% head floor in B2: stop and resolve. **P2:** the top between-component-variance quintile holds ≥ 40% of Σ_c Δ(c)² for q0. |
| **R1.2** | Density and interface at S1-QQ: T8 intrinsic dimension of exact conditional samples for ζ ∈ {c, c+qid*, c+qid*+sids}; interface term per φ_iface; σ rule for S1 and S1-QQ. | main s0–s4 | φ_iface = the first set with interface ≤ 2 SE. The h-space classifier at S1-QQ runs unsmoothed only if T8 ≥ 88 at that ζ; otherwise at σ. If the interface term with all IDs is still > 0.1·KL_model, the QQ tokens coordinate with the rest beyond IDs and c: a finding, carried into q1f. |
| **R1.3** | Ladder at S1 (full cut) and S1-QQ: R_marg → q1f, cumulative and add-one-to-q0. Δ, Δ_rms, Δ_rms^(c,IDs), D_f with floor, fragility control, DL, C(q). Also q0−ID once. | main s0–s4 (split by motif); untrained controls | **P3, motif:** in s0, s2, s3, Δ_rms(q0) ≥ 3·Δ_rms(q0+q1a); in s1, s4, ≤ 1.5×. **P4:** the Δ_rms reduction from q1b (tilted pools) is ≥ 2× larger in nats in one-pool than in two-pool seeds. **P5:** q1c (ID Grams) lowers Δ_rms^(c,IDs) by ≥ 20% in ≥ 4/5 seeds. **V8:** untrained nets are closed by R_in. A miss of P3 or P4 by more than 2 SE means this method or B2 is wrong at that point: stop and resolve before R1.4. If the fragility share is ≥ 0.5 for q0 at a token type, switch that type to the resample base. |
| **R1.4** | Oracle 2×2 at S1: q^R with S_a (QQ_s matched on (log S_s, o*_s) bins, independent per source), S_b (also o*_{−s}) and S_c (LAB matched on (r_sj, y_j)). Then the large-w loop (§7.4), ≤ 3 rounds on the best rung, at S1-QQ and S2a. | main s0–s4 | **P6:** in one-pool seeds S_a fails and S_b closes (reproducing the B2 MLP1 result). Output: per-seed (DL, Δ_rms) frontier, accepted named features, and each residual labelled L1–L5. |
| **R1.5** | Cross-layer consistency on the 2-layer model: tier i (identity) and tier ii, pushing the best S1 description through real block 2 → q̃ at S2a vs R1.1's q^fit. Then E5b, the 3-layer seeds: T2 → T3a with the same test, and the bridge at T3a on the regions found at S2a. | main s0–s4; L3_ws s10–14 | Pass: Γ_fit ≤ max(2 SE, 0.1·KL_model) and AUC ≤ 0.55. A fail at S1 → S2a with both sides closed = L5 (representation only), which becomes a hypothesis about block 2 for round 2. |
| **R1.6** (if time) | Contrasts and transfer: plateau net, seed 8, untrained; V9 transfer to seeds 5, 7, 9. | as listed | **P7:** the plateau net reaches S1 at S1 with a q conditioned on (Q, U), at DL ≥ 3× below the near-Bayes nets. Seed 8: which rung first closes S1-QQ, and if none, where the loop's features point (open question from B1). V9: structural transfer, Δ_rms ≤ 1.5×. |

**Round 2 (not planned in detail):** tier-iii surrogate chains (F̃_1, F̃_2, F̃_3) with DL_chain; the full 3-layer ladder at T1/T2; subset localisation (below) on the accepted rungs.

**Subset localisation, used in R1.4 and round 2.** A ranges over:
- token types;
- head outputs c_{l,h} (the attention output is x + Σ_h O_h c_h);
- the 64 MLP1 units at QQ, ranked;
- the top-4 relative-Fisher directions.

For each A:
- **Knock-out:** h_A ~ q(h_A \| h_{−A}, φ) by Gaussian conditioning, with the rest real. This gives Δ_A^out (structure in A that q cannot predict from the rest), and the interface split applies.
- **Knock-in:** h_A real, the rest from q. Δ_all − Δ_A^in is what A carries.
- **Interaction:** I = Δ_all − Σ_A Δ_A^out, and Shapley values over ≤ 4 groups.
- For MLP1: whether the top 8 units carry > 50% of Δ.

All are direct estimates with paired CIs.

**Across seeds:** report every seed. The seed mean is given with a t₄ interval, and episodes are never pooled across seeds or motifs.
