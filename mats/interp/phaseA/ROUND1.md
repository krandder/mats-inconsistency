# Round 1: replacing tiny-PFN layers by Bayes-latent Gaussians (2026-10-28)

**TL;DR.**
- **q0 is nowhere near closing.** q0 is a Gaussian whose mean is linear in the token's inputs plus the Bayes latents (per-source SS, variance posterior, precision weights β, query readings). Swapping it in for a layer costs L(0) − L(1) = 0.08–0.39 nats at the main read sites (full cuts, query-question tokens, the final query token). That is 19–81× the model's own KL to Bayes (0.003–0.013).
  - The latents are the right variables: they cut the cost from 4–5 nats (no latents) to 0.1–0.4.
  - But a linear Gaussian of them is far from the network. Only sites that downstream barely reads come near ε = 0.1·KL.
- **About half of that gap is off-manifold noise, not missing structure.** This is a post-hoc fragility control (METHOD §4.2): add q0-shaped noise to the *real* states.
  - That alone costs 33–37% of Δ(q0) at the 2-layer block-1 cut, and 47–65% at the 3-layer sites and the final query token.
  - An h-space classifier tells q0 states from real ones at AUC ≈ 0.999 everywhere.
  - So L(0) − L(1) mixes "structure beyond q" with how the network reacts off its manifold. The Gaussian base is the wrong family wherever that share is ≥ 0.5.
- **Where the gap sits.** Two localisations agree:
  - Keeping the *MLP output* real at the first block's cut recovers 98–100% of the gap in all 11 nets. The attention outputs are predictable from phi plus the rest.
  - Keeping 4 output-relevant directions per token real recovers 92–98% at QQ tokens and mid-network query tokens. A random 4-d subspace recovers about half.
- **Naming from where ŵ is large, one refinement step.** Two structures are named:
  - **(i) Query-question tokens (QQ)** depend on the *other* sources' query readings, but only in the one-pooling-head seeds. Adding them cuts Δ_rms by 35–48% in s0, s2, s3 and by 0% in s1, s4. This reproduces B2's motif split by a different method.
  - **(ii) The query token** depends nonlinearly, roughly quadratically, on the predictive mean and the latents. Pairwise products of the latents cut Δ_rms by 80–84% at the last cuts.
  - q1 (q0 plus the named group per token type) is accepted on fresh data at every full cut (16/16). It costs +4k to +13k bits under BIC, about 3–9% of one block at fp16 (137k bits). It is still 20–420× ε at the full cuts.
- **Kelvin's estimators.**
  - The output-space bridge L(0) − L(1) recovers 98.5% (median) of the direct gap wherever the classifier AUC ≥ 0.9, and is attenuated where AUC < 0.75.
  - The plain identity E_p G − E_q G = Cov_q(G, w) estimated from q samples is unusable here. ESS is 0.01–0.5% of draws at full cuts, and estimates range from 0.5× to 10⁶× the true gap.
  - q-side tempering with ŵ^α never reaches L(1). At α = 1 it sits at 0.03–0.11 against a true 0.005–0.010 (up to 17×).
- **Cross-layer consistency** (tier ii) fails in 16/16 pairs, as predicted for descriptions that are not near-closed. In the 3-layer nets, though, pushing the block-1 description through the *real* block 2 gives a better block-2 description (Δ 0.02–0.10) than fitting q0 at block 2 directly (0.16–0.52). The family at l+1 is the bottleneck, not the computation.
- **The plateau-stuck net.** It is described *better* by the binding-free q0U (unordered SS) than by q0: Δ_rms 0.19 vs 0.40 at QQ and 0.31 vs 0.40 at the full cut. The near-Bayes 2-layer nets go the other way at the full cut and the final query token (q0U 0.97–1.57× q0). Neither closes.
- **Pre-registered predictions.** P3, P5, P7 and P8 hold. P1, P2, P4 and P6 fail as stated; the details are below.

No decision is needed. The next step, round 2, is proposed at the end.

---

## 1. What was run (full spec: PREREG.md, written before any ladder result; amendments 1–3 are dated)

- **Nets.**
  - 2-layer w32_h2_L2 mix4 warm-start, seeds 0–4 (s1, s4: two pooling heads; s0, s2, s3: one pooling head plus a query mixer).
  - 3-layer L3_ws, seeds 10–14.
  - The plateau net w32_h2_L2_mix4_s3 (KL 0.146).
- **Data.**
  - Fit split: 20k episodes.
  - Eval A: 6,000 cells × 8 exact conditional draws (IDs, residual directions and labelled y resampled at fixed c = (Q, S)). Used for the ladder, ratios, localisation and naming.
  - Eval B: a fresh 6,000 × 8. Used for acceptance, DL and consistency.
  - CPU, 4 processes. About 1 h per 2-layer net and 1–2 h per 3-layer net.
- **G** = KL(p_B(·|c) ‖ f(h)) by quadrature; it matches `kl_grid` to 1e-15. Per-net KL on A matches eval.json to within 5%.
- **Sites.** (cut, token type), with the rest of the same draw real. hK = after block K; aK = after attention K.
  - Types: LAB (12 labelled tokens), QQ (3 query-question tokens), QT (the query token), and ALL (full cut).
  - Coordinates are shift-quotiented, and LayerNorm-scale-quotiented for the context tokens the last block reads.
- **q families.** LinearGaussian per token type: ridge mean and full residual covariance, with tokens independent given phi.
  - R_marg, R_in, **q0**, q0U (sorted SS plus the blind (Q, U) reference) and q0noID.
  - q0 at QT uses mean Q, the Bayes and labelled-only predictive moments, and Σβo*.
- **Estimators.**
  - Direct Δ = L(q) − L(p) (= Kelvin's L(0) − L(1)) with a cell bootstrap.
  - Δ_rms (per-cell RMS gap) from two independent halves.
  - Output-space ratio (MLP classifier on the 12 head outputs plus c, 2-fold cross-fit by cell, 2 refits). From it: the per-cell bridge L(α), Cov_q(G, ŵ) plain and self-normalised, E_q[Gŵ], Kish ESS, and checks T1, T2, V5.
- **Localisation.**
  - Residual-stream components [prev, head0, head1, mlp], modelled as a joint Gaussian whose sum is q0, with knock-out and knock-in.
  - 4-d subspaces: Uout (output-ratio gradient pulled back to h; Amendment 1), Uh (h-space classifier) and a random control.
- **Refinement.** Candidate groups were fixed in PREREG. q1 = q0 plus the group with the best held-out projected log-density gain along Uout, accepted on B if Δ_rms falls by ≥ max(10%, 3 SE).
- **DL** = struct bits + quantised param bits (tolerance 0.02·KL_model), plus BIC.
- **Consistency.** Tier i (identity) and tier ii (q̃ = q0 family refitted on F#q_l, vs q^fit fitted on real states).
- **Post hoc** (amendments 2–3): sub-splits of the winning candidates, and the fragility control.

## 2. The ladder: how much each description loses (eval A, direct, seed mean ± se)

Δ = L(q) − L(p) in nats. KL_model (seed mean) is 0.010 for 2L, 0.005 for 3L and 0.146 for the plateau net. ε = 0.1·KL_model.

| net | site | R_marg | R_in | **q0** | q0U (no binding) | q0noID | Δ_rms(q0) | Δ(q0)/KL |
|---|---|---|---|---|---|---|---|---|
| 2L | h1-LAB | 0.036 | 0.035 | 0.019 ± 0.009 | 0.023 | 0.019 | 0.035 | 1.5 |
| 2L | h1-QQ | 1.51 | 0.63 | **0.353 ± 0.018** | 0.454 | 0.354 | 0.637 | 39 |
| 2L | h1-QT | 2.04 | 2.04 | 0.047 ± 0.016 | 0.047 | 0.047 | 0.088 | 4.3 |
| 2L | h1-ALL | 4.07 | 2.68 | **0.389 ± 0.026** | 0.523 | 0.389 | 0.657 | 41 |
| 2L | a2-QT | 4.98 | 4.99 | 0.331 ± 0.047 | 0.542 | – | 0.717 | 39 |
| 2L | h2-QT | 5.05 | 5.03 | **0.221 ± 0.026** | 0.488 | – | 0.526 | 25 |
| 3L | h1-QQ | 2.62 | 0.38 | 0.136 ± 0.062 | 0.276 | 0.183 | 0.259 | 26 |
| 3L | h1-ALL | 3.28 | 0.93 | 0.151 ± 0.059 | 0.337 | 0.187 | 0.281 | 29 |
| 3L | h2-LAB | 0.008 | 0.007 | 0.003 ± 0.002 | 0.003 | 0.003 | 0.004 | 0.4 |
| 3L | h2-QQ | 0.55 | 0.43 | **0.344 ± 0.077** | 0.259 | 0.349 | 0.452 | 78 |
| 3L | h2-ALL | 4.79 | 4.03 | **0.358 ± 0.062** | 0.480 | 0.355 | 0.450 | 81 |
| 3L | h3-QT | 5.32 | 5.32 | **0.083 ± 0.009** | 0.422 | – | 0.268 | 19 |
| plateau | h1-QQ | 0.30 | 0.27 | 0.246 | **0.103** | 0.247 | 0.396 | 1.7 |
| plateau | h1-ALL | 3.53 | 3.38 | 0.259 | **0.193** | 0.260 | 0.405 | 1.8 |
| plateau | h2-QT | 4.12 | 4.11 | 0.205 | **0.163** | – | 0.412 | 1.4 |

Per-seed values with CIs and Hill indices, and the rows not shown here, are in `r1/tables.md`.

**Plain findings.**
- **The Bayes latents carry the bulk of what the layer holds.** At the query token, Δ falls from about 5 nats (no latents) to 0.08–0.33. At QQ, adding the latents to R_in halves Δ.
- **q0 is still far from the network.** No read-heavy site comes within 10× of ε.
- **The sites that go below ε are not real closures.** They are LAB tokens downstream barely reads: h1-LAB in the two-pooling-head seeds s1 and s4 (Δ ≈ 0, so layer 2 reads only QQ there), plus h1-LAB, h1-QT and h2-LAB in 3L s13.
- **The 3-layer nets carry less residual than the 2-layer nets** at the last query-token cut (Δ_rms 0.27 vs 0.53) and at the main full cut (0.45 vs 0.66). The ratio to the 3-layer nets' smaller KL is still larger.
- **Linear ID terms do nothing.** q0noID equals q0 to within 2% at every site. After block 1, downstream does not route through linearly encoded IDs given the latents. The 22% "ID crosstalk" in B2 must enter nonlinearly.

## 3. Kelvin's quantities: L(α), L(0) − L(1), Cov_q(G, w), ESS (q0, output-space ratio)

Seed means. "bridge/direct" = (L(0) − L(1)) from the classifier, divided by the direct Δ. ESS_q is the Kish ESS of ŵ over q draws, as a fraction of all draws.

| net | site | AUC | direct Δ | L(0)−L(1) (bridge) | bridge/direct | −Cov_q(G,ŵ) plain | −Cov_q SNIS | E_q[Gŵ] − E_pG (should be 0) | ESS_q | T1 / T2 / V5 pass |
|---|---|---|---|---|---|---|---|---|---|---|
| 2L | h1-LAB | 0.72 | 0.019 | 0.016 | 0.55 | 0.010 | 0.012 | 0.012 | 0.44 | 2 / 2 / 0 of 5 |
| 2L | h1-QQ | 0.97 | 0.353 | 0.350 | 0.991 | 0.36 ± 0.09 | 0.262 | 0.017 | 0.008 | 5 / 0 / 5 |
| 2L | h1-ALL | 0.98 | 0.389 | 0.385 | 0.991 | 0.95 ± 0.44 | 0.287 | 0.066 | 0.001 | 5 / 0 / 3 |
| 2L | h2-QT | 0.97 | 0.221 | 0.217 | 0.979 | 0.32 ± 0.11 | 0.154 | 0.071 | 0.004 | 5 / 0 / 5 |
| 3L | h1-QT | 0.70 | 0.013 | 0.011 | 0.58 | 0.008 | 0.009 | 0.002 | 0.44 | 5 / 1 / 4 |
| 3L | h2-QQ | 0.98 | 0.344 | 0.342 | 0.992 | 66 ± 61 | 0.286 | 37 ± 35 | 0.0005 | 5 / 0 / 1 |
| 3L | h2-ALL | 0.99 | 0.358 | 0.353 | 0.979 | 9.1 ± 5.1 | 0.283 | 0.77 | 0.0004 | 5 / 0 / 1 |
| 3L | h3-QT | 0.96 | 0.083 | 0.076 | 0.910 | 0.040 | 0.034 | 0.071 | 0.002 | 5 / 0 / 5 |
| plateau | a2-QT | 0.99 | 0.227 | 0.220 | 0.972 | 2.5e5 | 0.106 | 1.1e5 | 0.0001 | 1 / 0 / 1 |

**L(α) curves** (bridge, seed mean; α = 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1). For comparison, Kelvin's q-side self-normalised tempering L_q(α) = Σ ŵ^α G / Σ ŵ^α over the q draws:

| net, site | bridge L(α) | q-side L_q(α) | true L(1) |
|---|---|---|---|
| 2L h1-ALL | 0.396, 0.159, 0.078, 0.047, 0.024, 0.015, 0.0105 | 0.399, 0.195, 0.147, 0.130, 0.118, 0.114, 0.112 | 0.0101 |
| 2L h2-QT | 0.228, 0.086, 0.044, 0.030, 0.019, 0.014, 0.0109 | 0.232, 0.121, 0.096, 0.087, 0.081, 0.079, 0.078 | 0.0101 |
| 3L h2-ALL | 0.395, 0.104, 0.053, 0.032, 0.015, 0.009, 0.0055 | 0.398, 0.129, 0.098, 0.089, 0.083, 0.081, 0.080 | 0.0048 |
| plateau h1-ALL | 0.401, 0.343, 0.305, 0.269, 0.209, 0.172, 0.150 | 0.405, 0.357, 0.340, 0.334, 0.328, 0.327, 0.326 | 0.146 |

**Plain findings.**
- **When q is far from p (AUC ≥ 0.9, 64 site-nets), the pooled bridge is accurate.** Its endpoints match the direct means (T1 passes at 60/64), and L(0) − L(1) recovers 98.5% of Δ (median; range 83–100%).
- **The ∫dL/dα check (V5) passes at only 46/64.** With 11 grid points, the steep drop near α = 0 is under-resolved, so use the endpoint difference, not the integral.
- **The attenuation VALIDATION.md found appears only where q is close in output space** (AUC 0.6–0.75): there the bridge recovers 34–69% (P5).
- **Kelvin's q-side identity, Cov_q(G, w) estimated from q samples with plain weights, does not work at these sites.**
  - The ESS is 1–47 draws per 10,000 at full cuts, and k̂ is 0.6–1.6.
  - The estimate ranges from 0.5–2.4× Δ (2L, 3L QT sites) to 25–10⁶× Δ (3L QQ and full cuts, the plateau net). E_q[ŵ] ≠ 1 (T2 fails at every AUC ≥ 0.9 site).
  - The self-normalised version is stable but biased: it gives 41–83% of Δ.
- **q-side tempering stalls.** Its α = 1 end is up to 17× the true L(1). Real-like draws are absent from q's sample, so no reweighting can reach them.
- **Bottom line:** report L(0) − L(1) from the pooled bridge, or directly. The variance warning in Kelvin's plan (E_q[w²] = 1 + χ²) is the binding one here, and the q-side forms should only be diagnostics.

**The L(α) shape.**
- **Near-Bayes nets:** most of the drop happens by α = 0.2. A small amount of real-structure weight removes most of the gap, which fits a gap concentrated in a few directions (§4).
- **Plateau net:** the curve is close to linear in α.

## 4. Where the gap lives (localisation, eval A, q0)

**Residual-stream components.** carried(C) = Δ_all − Δ(knock-in C) is what keeping C real recovers, as a fraction of Δ_all of the q0-equivalent stacked Gaussian.

| net, cut | Δ_all | carried: prev | head0 | head1 | MLP |
|---|---|---|---|---|---|
| 2L h1, s0 (mixer = h0) | 0.41 | 0.20 | **0.48** | 0.19 | **0.99** |
| 2L h1, s2 (mixer = h1) | 0.47 | 0.20 | 0.25 | **0.60** | **0.99** |
| 2L h1, s3 (mixer = h1) | 0.39 | 0.22 | 0.22 | **0.46** | **0.99** |
| 2L h1, s1 / s4 (two pooling heads) | 0.31 / 0.36 | 0.33 / 0.36 | −0.01 / 0.00 | 0.02 / 0.01 | **0.99 / 0.99** |
| 3L h1, 5 seeds | 0.08–0.39 | 0.00–0.02 | 0.02–0.22 | 0.02–0.73 | **0.98–1.00** |
| 3L h2, 5 seeds | 0.16–0.52 | 0.07–0.65 | 0.33–0.80 | 0.03–0.70 | 0.75–0.99 |
| 2L h2 (QT), 5 seeds | 0.17–0.30 | 0.40–0.61 | 0.61–0.89 | 0.71–0.82 | 0.70–0.94 |
| 3L h3 (QT), 5 seeds | 0.07–0.11 | **0.71–0.92** | 0.36–0.75 | 0.41–0.70 | 0.52–0.87 |
| plateau h1 | 0.26 | 0.26 | 0.37 | 0.66 | **1.00** |

- **At the first post-block cut, everything q0 misses is in the MLP's write.** Given the real MLP output and phi, the embedding and both attention outputs are Gaussian-predictable in all 11 nets.
- **The heads split by motif.** In the two-pooling-head seeds the attention heads carry nothing beyond q0. In the one-pool seeds the *query-mixer head* carries 0.46–0.60 and the pooling head 0.19–0.25. That is the mixer head bringing other sources' readings into QQ (B2).
- **At the last cut of the 3-layer nets, the gap is inherited.** It sits mostly in the residual stream coming into block 3 ("prev" carries 0.71–0.92), not in block 3's own writes.

**4-d subspaces.** Δ remaining when only 4 dims are kept real (knock-in), as a fraction of the site's Δ(q0). Smaller means those 4 dims carry the gap.

| net | site | Uout (G-relevant) | Uh (h-space classifier) | random 4-d | h-space AUC |
|---|---|---|---|---|---|
| 2L | h1-QQ | **0.025** | 0.33 | 0.34 | 0.999 |
| 2L | h1-QT | 0.080 | 0.21 | 0.45 | 0.999 |
| 2L | h2-QT | 0.16 | 0.35 | 0.51 | 0.999 |
| 3L | h1-QQ / h2-QQ | **0.030 / 0.021** | 0.26 / 0.48 | 0.43 / 0.50 | 0.998 |
| 3L | h2-QT | **0.018** | 0.41 | 0.58 | 0.999 |
| 3L | h3-QT | 0.70 | 0.24 | 0.49 | 0.998 |
| plateau | h1-QQ / h2-QT | 0.018 / 0.47 | 0.54 / 0.33 | 0.10 / 0.52 | 0.999 |

- **At QQ tokens and mid-network query tokens, 4 output-relevant directions carry 97–98% of the gap.**
- **The h-space classifier's directions are G-irrelevant**, no better than random. That is why Amendment 1 switched to the pulled-back output ratio.
- **The exception is the final query-token cut of the 3-layer nets** (and the plateau net's final cut). There 4 directions are not enough; the 12-output head reads more of the state.

**Where in episode space.**
- **At the late query-token cuts (after attention or the block in the last layer, and 3L h2), the q0 gap is *not* concentrated in multimodal episodes.** The top between-component-variance quintile holds 15–20%, at or below the uniform 20%.
- **It is concentrated in the top decile of log ŵ_out** (28–42%).
- **At QQ it leans mildly toward multimodal and high-coupling episodes** (26–33% per top quintile).

## 5. Refinement: naming the missing structure, q1, and its cost (fresh eval B)

**Where ŵ_out is large (top decile), the library features shift as follows** (z-scores of the top decile against all draws, consistent across seeds):
- **QT** (h2 of 2L, h2/h3 of 3L): the top features are all *squared predictive means*: Bayes mean², (Σβo*)², mean × Σβo*, labelled-only mean × Σβo*, all at +0.9 to +1.5 sd. So the real query-token state bends with |predictive mean| in a way a linear map cannot follow.
- **QQ:** high between-component variance (+0.3 to +1.8), high coupling KL(w_B ‖ w_lab) (+0.3 to +1.3), *negative* dE_s = E_B,s − E_lab,s (−0.2 to −1.2), and in two-pool seeds a large own reading o*_s² (+1.0). These are episodes where the query readings lower that source's apparent reliability.

**q1 per site, on fresh B** (seed mean ± se). Δ_rms is the per-cell RMS gap.

| net | site | q1 adds (most common) | Δ_rms q0 → q1 | reduction | accepted | Δ q0 → q1 | q1 / ε |
|---|---|---|---|---|---|---|---|
| 2L | h1-QQ | cross (s0, s2, s3); quadlat (s1, s4) | 0.64 → 0.36 | 43% | 5/5 | 0.352 → 0.178 | 380 |
| 2L | h1-ALL | per type | 0.66 → 0.39 | 41% | 5/5 | 0.390 → 0.204 | 420 |
| 2L | h2-QT | quadlat | 0.50 → 0.096 | 81% | 5/5 | 0.206 → 0.038 | 100 |
| 3L | h1-ALL | per type | 0.28 → 0.060 | 79% | 5/5 | 0.149 → 0.047 | 113 |
| 3L | h2-QQ | cross | 0.46 → 0.22 | 53% | 5/5 | 0.344 → 0.139 | 470 |
| 3L | h2-ALL | per type | 0.45 → 0.18 | 61% | 5/5 | 0.356 → 0.149 | 375 |
| 3L | h3-QT | quadlat | 0.26 → 0.049 | 80% | 5/5 | 0.075 → 0.016 | 110 |
| plateau | h1-ALL / h2-QT | episode / betaQ | 0.40 → 0.30 / 0.40 → 0.22 | 25% / 44% | yes | – | 20 / 15 |
| 2L, 3L | LAB sites | r2 / pool | – | 4–15% | 3/15 | – | – |

- **The naming rule picks well.** On B it chose the best add-one candidate in 47/68 site-nets, and a candidate reaching ≥ 80% of the best reduction in 56/68.
- **POST HOC sub-splits (Amendment 2), Δ_rms on B, seed means.**
  - *QQ, 2L h1:* q0 0.64; +other sources' readings and SS (cross_o) 0.53; +query-coupled own posterior (cross_p) 0.62; both 0.52.
    - Per seed: one-pool s0, s2, s3 go 0.48 → 0.34, 0.59 → 0.35 and 0.48 → 0.34 with cross_o.
    - Two-pool s1, s4: 0.76 → 0.74 and 0.90 → 0.90 (nothing).
    - The missing QQ structure in one-pool seeds is *the other sources' readings*, carried in by the mixer head, not the coupled posterior.
  - *QQ, 3L h2:* 0.46; cross_o 0.24; cross_p 0.41; both 0.21. The same holds in every 3-layer seed.
  - *QT:* five squared-mean features (meansq) get most of the way. 2L h2: 0.50 → 0.17 (all pairwise latent products 0.096). 3L h3: 0.26 → 0.064 (0.050).
- **Description length.**
  - q0 at a full cut costs 34–56k bits; one block at fp16 is 137k bits.
  - q1 adds −2k to +23k quantised bits per full cut. The quantised DL is noisy: the bit depth b* swings between 2 and 14 because the tolerance (0.02·KL) is tiny next to Δ.
  - BIC, which moves monotonically, adds +4k to +13k bits.
  - Break-even N* (two-part code) is 1×10⁴ to 1.5×10⁵ episodes per accepted rung.
  - q1 is already at 36–50% of a block, so METHOD's S3 budget (0.5 × block = 68k bits) leaves room for at most one more rung of this size at a full cut.
- **The cost of naming.** With q1 the residual is still 20–470× ε at the full cuts and QQ sites. One step of "add the named feature linearly" removes 40–85% of Δ_rms, not 99%.

## 6. Fragility control (POST HOC, METHOD §4.2; eval B)

Δ in nats, seed mean. "Noise share" = Δ(real states + q0's residual noise) / Δ(q0). METHOD's rule: a share ≥ 0.5 means the Gaussian base is the wrong family for that site.

| net | site | Δ q0 | Δ real + noise | noise share (per seed) | Δ q0 mean only (no noise) |
|---|---|---|---|---|---|
| 2L | h1-ALL | 0.389 | 0.141 | 0.37 (0.34–0.41) | 0.262 |
| 2L | h1-QQ | 0.352 | 0.117 | 0.33 (0.30–0.36) | 0.249 |
| 2L | h1-QT | 0.045 | 0.025 | 0.58 (0.52–0.65) | 0.020 |
| 2L | h2-QT | 0.206 | 0.098 | 0.48 (0.45–0.51) | 0.102 |
| 3L | h2-ALL | 0.356 | 0.234 | **0.65** (0.49–0.84) | 0.142 |
| 3L | h2-QQ | 0.344 | 0.175 | 0.50 (0.38–0.61) | 0.222 |
| 3L | h2-QT / h3-QT | 0.141 / 0.075 | 0.074 / 0.035 | 0.52 / 0.47 | 0.064 / 0.039 |
| plateau | h1-ALL / h2-QT | 0.255 / 0.199 | 0.167 / 0.145 | **0.66 / 0.73** | 0.125 / 0.048 |

- **The q0 gap has two comparable parts.**
  - A wrong *mean*: q0's mean alone, with no noise, loses 0.10–0.26 nats.
  - The network's sensitivity to Gaussian noise of q0's residual shape: 0.1–0.23 nats even on top of the *real* states.
- **In the 2-layer nets at QQ/full cut the mean dominates** (noise share about 1/3), so the naming loop is aimed at the right thing there.
- **Elsewhere the share is about 0.5 or more** (3L full cut, QT sites, plateau net). Much of "structure beyond q0" there is the network extrapolating off-manifold, which a Gaussian base will always provoke. That supports METHOD's switch to an on-manifold (resampling) base for those token types.
- This is a caveat on reading L(0) − L(1) as "loss attributable to structure in layer l". With a Gaussian q it is an upper bound that includes off-manifold response.

## 7. Cross-layer consistency (eval B)

- **Tier i** (Δ_l(q_l) = Δ_{l+1}(F#q_l), sample by sample): exact (max |ΔG| = 0) in all 16 pairs.
- **Tier ii.** q_l = the accepted q1 at the full cut. q̃ = the q0 family refitted on states pushed through the real next block; q^fit = q0 fitted on real states.

| pair | Δ(F#q_l): q_l pushed through the real block | Δ(q̃): that, re-projected to the q0 family | Δ(q^fit) | Γ_fit = Δ_rms(q̃) − Δ_rms(q^fit) | AUC(q̃ vs q^fit) | pass |
|---|---|---|---|---|---|---|
| 2L h1 → h2 (5 seeds) | 0.16–0.23 | 0.29–0.34 | 0.16–0.28 | +0.07 to +0.17 (SE ~0.012) | 0.93–0.97 | 0/5 |
| 3L h1 → h2 (5 seeds) | **0.02–0.10** | 0.17–0.61 | 0.16–0.52 | −0.001 to +0.10; within 2 SE in 4/5 | 0.71–0.98 | 0/5 |
| 3L h2 → h3 (5 seeds) | 0.05–0.31 | 0.14–0.36 | 0.06–0.10 | +0.015 to +0.22 | 0.97–0.99 | 0/5 |
| plateau h1 → h2 | 0.19 | 0.31 | 0.20 | +0.12 | 0.98 | 0/1 |

- **Every pair fails Kelvin's condition (P7).** The descriptions are not near-closed, so this mainly says "q0-family descriptions at adjacent layers are not related by the real block".
- **In the 3-layer nets, the block-1 description pushed through the real block 2 is a much better block-2 description than any q0 fitted at block 2** (0.02–0.10 vs 0.16–0.52). Re-projecting it onto the Gaussian family throws that away: Γ_proj = 0.14–0.58.
- So at h2 the *computation* applied to a crude h1 description beats a direct description; the q0 family is the bottleneck. Tier iii (short surrogate blocks F̃) is the natural form here.

## 8. The plateau-stuck net vs the near-Bayes nets

- **Binding-free q0U vs q0 (Δ_rms ratio q0U / q0):**
  - Plateau net: 0.48 at h1-QQ, 0.77 at h1-ALL, 0.77 at h2-QT. Forgetting which SS belongs to which source *improves* the description.
  - Near-Bayes 2L: 0.83–1.28 at h1-QQ, 0.97–1.34 at h1-ALL, 1.30–1.57 at h2-QT.
  - This matches astra's weak-binding result from another angle: the stuck net's layer-1 QQ states look like functions of the unordered reliabilities.
- **Neither family closes the plateau net.** Δ_rms(q0U) = 0.31 at the full cut against ε = 0.015. So the pre-registered "no structure beyond unordered reliabilities" (P2) fails.
- **The plateau net's gap is the most noise-driven** (fragility share 0.66–0.73). Its MLP1 carries all of the block-1 gap, and its refinement is weakest (18–44%).

## 9. Pre-registered predictions

| | Prediction | Outcome |
|---|---|---|
| P1 | q0 does not close: Δ_rms > ε at every site, full-cut Δ > KL_model | **Fails as stated** (7/10 nets): barely-read LAB sites fall below ε (2L s1/s4 h1-LAB; 3L s13 h1-LAB, h1-QT, h2-LAB). The substantive part holds: full-cut Δ > KL in 10/10, and every read-heavy site is ≥ 10× ε. |
| P2 | Plateau: q0U ≈ q0 and q0U closes; near-Bayes: q0U ≥ 2× q0 | **Fails.** The direction holds (plateau q0U ≤ q0 at all 3 sites; near-Bayes q0U worse at the full cut and QT), but the plateau net does not close (0.31 vs 0.015), and the near-Bayes ratios are 1.0–1.6, not ≥ 2. |
| P3 | 3L smaller residual than 2L | **Holds:** last QT 0.27 vs 0.53, full cut 0.45 vs 0.66. |
| P4 | One-pool h1-QQ Δ ≥ 2× two-pool, and naming picks `cross` | **Fails** on magnitude (0.31–0.38 vs two-pool mean 0.37). Naming picks `cross` in 3/3 one-pool seeds, and post hoc `cross` helps *only* in one-pool seeds (35–48% vs 0%). |
| P5 | Bridge attenuated where AUC < 0.6 | **Holds** (6/6 sites); at AUC ≥ 0.9 the bridge is unbiased to about 1.5%. |
| P6 | Dropping IDs doubles Δ at h1-ALL | **Fails:** the ratio is 0.99–1.02. Linear IDs are not used downstream given the latents. |
| P7 | Tier i exact; tier ii fails for 2L h1 → h2 | **Holds** (5/5; 16/16 overall). |
| P8 | q1 accepted at 2L h1-ALL in ≥ 3/5, still > ε | **Holds** (5/5 accepted, all still > ε). |

## 10. What remains unexplained

1. **Most of the q0 gap.** After one refinement, Δ_rms at the main full cut is 0.39 (2L) and 0.18 (3L) nats, still 100–500× ε. Splitting the remaining gap between wrong mean and off-manifold noise:
   - 2L QQ: the mean dominates.
   - 3L full cut and QT sites: about half is noise response.
2. **QQ in the two-pooling-head seeds** (s1, s4).
   - None of the cross-source features help.
   - Quadratic own-source latents cut it about 50%, to 0.40–0.45.
   - The MLP1 output is still the carrier. Its nonlinear code (B2's "nonlinear part of the reliability code") has no short name yet.
3. **The final readout at the 3-layer h3 query token.** 4 G-relevant directions recover only 30% there. The gap is inherited from block 2's stream ("prev" carries 0.71–0.92), so it should be named at h2, not h3.
4. **LAB tokens at 2L h1.** The gap is small (0.019) and no candidate moves it (0–7%).
5. **ID crosstalk.** It is invisible to q0 because q0 is linear in the IDs; the 22% budget share must be nonlinear.
6. **The plateau net.** Neither the bound nor the unbound Gaussian fits. Its states are dominated by off-manifold sensitivity under Gaussian noise.

## 11. Caveats and deviations

- **Scope cuts** (stated in PREREG before running): no σ-smoothing, no DeepSets classifier, no uLSIF or quadratic family on outputs, no oracle q^R, no untrained, transfer or seed-8 runs, no tier-iii chains, and no M ∈ {8, 16, 32} SNIS-bias check.
- **Classifier settings and q0 content.** The output-space classifier uses 2 refits only. q0 includes β_s, as the task asked; that is a deviation from METHOD §7.3.
- **Amendment 1 (naming directions).** It was made after a smoke run at n = 800, before any full run: the naming directions come from the output ratio pulled back to h, not from the h-space classifier. Both are reported; the naming choice agreed between them at most QT sites.
- **Amendments 2–3 are post hoc** (candidate sub-splits; the fragility control) and are labelled as such everywhere.
- **Heavy tails.** The Hill index of per-cell differences is < 2 in many cells (3L a3/h3-QT, h1-QQ), so the CIs there under-cover (VALIDATION finding 7). The seed spread, not the within-seed CI, is the honest error bar.
- **Quantised DL is noisy** (b* 2–14). Orderings use BIC, which agrees in direction for every accepted rung.

## 12. Proposed round 2 (not run; bounded)

- **(a) On-manifold base at the sites with noise share ≥ 0.5:** the 3L full cut, the QT sites and the plateau net. Use conditional resampling of real token states within feature bins (METHOD q^R), so that L(0) − L(1) measures structure rather than extrapolation.
- **(b) Name the 3-layer h3 residual at h2,** where it originates.
- **(c) Tier iii:** replace block 2 by a short surrogate F̃ acting on the h1 description. The tier-ii result says the pushforward through the real block is already the better h2 description.

## Files (`/home/agent/workspaces/mats/work/cm-interp/`)

- **Pre-registration:** `PREREG.md` (predictions and amendments 1–3, timestamped).
- **Code:**
  - `r1run.py`: the per-net pipeline.
  - `lib/tp.py`: cuts, exact conditional sampler, features, G quadrature.
  - `lib/r1est.py`: estimators, q wrapper, subspace knock-out.
  - `r1posthoc.py` and `r1fragility.py`: the post-hoc runs.
  - `r1summ.py`: builds the tables.
- **Results:**
  - `r1/<net>.json`: everything per net.
  - `r1/posthoc_<net>.json` and `r1/fragility_<net>.json`.
  - `r1/tables.md`: all tables, including per-seed values.
  - `r1/predictions.md`.
  - `r1/logs/`.
- **Smoke runs** at n = 800, kept for the record: `r1/*_quick.json`.
