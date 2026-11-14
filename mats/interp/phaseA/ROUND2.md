# Round 2: information-matched, on-manifold oracles (2026-10-28)

**TL;DR.**
- **What was run.** A site's tokens are replaced by the network's *real* states computed on fresh world episodes drawn
  given exactly what a description may read ("oracles"). Nothing is fitted and every donor state is a real state.
  11 nets, every round-1 site, fresh data, the weight-free direct estimator only. Pre-registered (PREREG amendments
  4–5) before any full-size result; amendment 6 is a forced post-hoc repair, stated below.
- **Controls pass on Δ.** One joint donor with c and all IDs (J) at every full cut, and O_c at every final cut, give
  Δ = 0 within 2 SE in all 11 nets, so the samplers are exact and the estimator is unbiased. The other half of the
  control, Δ_rms ≤ ε, cannot be resolved at n = 6,000: the null's upper bound is 0.0006–0.005 nats, above ε for every
  near-Bayes net.
- **Final query-token cuts in the 10 near-Bayes nets: q0's gap is all form, and the form is mostly the conditional
  mean.**
  - The oracle with q0's information (O_q0m) loses ≤ 0.002 nats (Δ_rms 0.006–0.009) against q0's 0.06–0.46
    (Δ_rms 0.19–0.60).
  - q0's own family (Gaussian, homoscedastic) with the true conditional mean (M_q0m) loses 0.004–0.015, so 90–98% of
    q0's gap is its linear mean.
  - The plateau net differs, and only Δ_rms shows it. Its signed oracle gap is 0.002, but its Δ_rms is 0.12, 31% of
    q0's: q0's six Bayes-oriented features miss per-cell information that the plateau's readout uses, and the errors
    cancel in the mean (P9 fails there).
- **Query-question (QQ) tokens and full cuts split by what layer-1 attention reads.**
  - *Own-source regime:* 2L two-pool nets (s1, s4) and the 3L nets after block 1 (except s11). The oracle with q0's
    information loses ≤ 0.002 at QQ tokens against q0's 0.04–0.38. q0's gap there is ~100% form; q0's own family with
    the true conditional mean recovers 74–98% of it.
  - *Cross-source regime:* 2L one-pool nets (s0, s2, s3), 3L s11 at both cuts, every 3L net after block 2, and the
    plateau net after block 1. q0 lacks the other sources' (Q, S). Here the split is order-dependent:
    - changing form first leaves the oracle almost as bad as q0: form is −8% to 44% of Δ(q0), median 10%; the
      cross-source variables are 45–106%;
    - adding the variables first, linearly (q0x), gives form 14–82%, median 43%;
    - Shapley gives form 8–63%, median 27%.

    (20 site-nets.)

    What does not depend on the order: once c is given, the remaining form gap is again the mean (M_c is within
    0.000–0.018 of O_c, while q0x, linear in the same variables, loses 0.04–0.36).
- **What c itself does not carry** (oracle ladder):
  - beyond c, one-pool QQ tokens still lose 0.007–0.024 nats; adding the own source's residual vector and labels
    (O_cIDr) removes almost all of it (≤ 0.0003 left);
  - other tokens' IDs: ≤ 0.015 nats everywhere;
  - in 3L s11 and s14 after block 2, cross-token consistency of nuisance variables: 0.05–0.07 nats with per-token
    draws, 0 with one joint draw. No per-token-independent family can capture this; it needs shared cross-token factors.
- **Binding.**
  - *Whole input, no stitching (astra's measure):* B_JS = 0.91–1.02 in the 10 near-Bayes nets and 0.012 in the plateau
    net, whose output ignores the source assignment.
  - *Per layer:* a binding-free description would lose 0.05–0.37 nats in near-Bayes nets and nothing in the plateau
    net. In 3L s12 binding happens only in block 2: the layer term is 0.000 after block 1 and 0.23 after block 2, and
    the token's own IDs cost 0.26 after block 1.
- **Stitching (astra's caution) is real but mostly G-irrelevant.**
  - In the near-Bayes nets a whole-layer classifier separates stitched from real layers at AUC 0.93–1.00 for every
    per-token oracle short of "c + all IDs + own residuals"; only the joint donor at a full cut sits at 0.50.
  - In the plateau net, QQ tokens drawn given c are undetectable (AUC 0.50).
  - Yet wherever the information suffices (own-source regime), those same detectably stitched layers cost ≤ 0.002
    nats.
- **Readout control** (estimator review, fix 2): see §6. Keeping the top-4 ∇G directions or the MLP write real removes
  almost all of R_in's gap as well as q0's. So round 1's "4 directions carry the gap" and "the MLP write carries the
  gap" describe the readout, not q0.
- **Next family:** a per-token Gaussian with homoscedastic noise and a **nonlinear conditional mean**. Its inputs are
  own-source latents, plus the other sources' (Q, S) where attention mixes sources, plus the own residual vector, plus
  shared cross-token factors at 3L h2. Stop naming linear features. Shape machinery (resampling, mixtures, flows) is
  not needed at these sites.

No decision is needed from Kelvin; the next step (a nonlinear-mean family on the ladder's variables) is proposed in §8.

---

## 1. What was run (PREREG.md amendments 4–6)

- **Nets and sites:** the 11 nets and every site of round 1:
  - 2L s0–s4, 3L s10–s14 and the plateau net;
  - LAB, QQ, QT and full (ALL) cuts after each non-final block;
  - the query token after the last attention and after the last block.
- **Data:**
  - Fresh eval E: 6,000 cells (seed 70000) × 8 exact conditional draws (70001).
  - q0, q0x (= q0 + `cross`) and q1 are refit on round 1's fit split (seed 60000), so q0 is round 1's q0. Its fresh Δ
    reproduces round 1 within noise (2L s0 h1-ALL 0.391 vs 0.403).
  - Every variant is paired with the same real draws (common random numbers).
- **Oracles.**
  - For each eval draw x' and each token t of a site, a fresh episode x''_t ~ p(x | ψ_t(x')) is drawn from the world.
    Token t is replaced by the network's state h_t(x''_t), and the rest of x' stays real.
  - Tokens are drawn independently, as q0 assumes. raw_t is the token's own IDs and values.

  | oracle | ψ_t (held fixed) | role |
  |---|---|---|
  | O_lat | c = (Q, S) only, IDs fresh | "Bayes latents only" |
  | O_own | raw_t + own-source (Q_s, S_s) | q0 minus β_s |
  | **O_q0** | O_own + β_s (32 exact candidates, nearest β; error median 0.003 against a conditional sd of 0.11) | **exactly what q0 reads** at LAB/QQ |
  | O_q0m | query token: nearest neighbour of q0's six features among 4,000,000 prior episodes (0.027 sd) | what q0 reads at QT (amendment 6) |
  | O_q0ID | O_q0 + all IDs | order check |
  | O_c | raw_t + c | = what q0x reads |
  | O_U | raw_t + (Q, unordered S); assignment drawn from its exact posterior (P(identity) = 0.27 on average) | binding |
  | O_cID | raw_t + c + all IDs | |
  | O_cIDr | O_cID + own source's residual vector + labelled y | |
  | J | one draw for the whole site: c + all IDs | V6 control |
  | M_q0, M_c | mean of 8 O_q0 (O_c) draws + N(0, (1 − 1/8) Σ_w), Σ_w = pooled within-ψ covariance | q0's family with the true conditional mean |

- **Samplers.**
  - Exact unit tests (`tests/test_oracle.py`, 5/5): kept variables are exact, x'' ~ p, and (x', x'') is exchangeable
    at 200k draws.
  - Two samplers are approximate, and their errors are reported: the β match, and the query-token nearest-neighbour match.
- **Estimators:** direct and weight-free only.
  - Δ with a studentised cell bootstrap; Δ_rms from two independent halves; Hill index; 1%-trimmed mean.
  - Shares are taken of the signed Δ along fixed nested orders, never as differences of Δ_rms (§3.3).
- **Coordinator additions (astra):**
  - a whole-layer real-vs-stitched classifier (§5);
  - astra's stitching-free whole-input binding measure (§4): the posterior-weighted Jensen–Shannon divergence of the
    output over the six SS-to-source assignments, with full Bayes as the positive and unordered Bayes as the null
    control.
- **Estimator-review fixes:**
  - no L(α) interior is reported, only its endpoints (the direct means);
  - the round-1 localisations are re-run on all 11 nets with readout controls (§6).
- **Cost:** about 30 min per 2L net and 45 min per 3L net on one CPU process, 4 processes.

## 2. Controls

| check | result |
|---|---|
| C1, V6 on Δ (J at full cuts; O_c at final query-token cuts) | \|Δ\| ≤ 2 SE in **38/38** site-nets (largest \|z\| 1.38; largest \|Δ\| 0.0003 nats) |
| C1, V6 on Δ_rms ≤ ε | **not resolvable** (passes 3/38, only in the plateau net, whose ε is 15× larger): the null's upper 95% bound is 0.0006–0.005 nats against ε = 0.0003–0.0015. Pre-registered as a stop rule; it fails by resolution, not by bias (point estimates 0.000–0.003), so no ε-level closure claim is possible at n = 6,000 for any description |
| C2, whole-layer classifier on J at the full cut | AUC 0.498–0.505 in 11/11 nets |
| C3, JS null (unordered Bayes) and quadrature | JS_blind ≤ 4e-16; 801- vs 4,001-point grids agree to 3e-13 |
| C4, β match | median error 0.0031, 99th percentile 0.045, in every net |
| Premise for O_q0 = O_c at the query token (amendment 4) | **fails**: q0's six features recover the sorted log S with R² 0.64–0.79 only. It was repaired by O_q0m (amendment 6). In signed Δ the repair barely matters (O_q0m − O_c ≤ 0.002 nats at every query-token site). In Δ_rms it does not matter in the near-Bayes nets (0.006–0.009), but it does in the plateau net (0.12 vs 0.003) |

## 3. Decomposition of q0's gap

### 3.1 Final query-token cuts: all form, mostly the mean

Signed Δ in nats; seed mean [range over seeds]. O_q0m = the oracle with exactly q0's information (amendment 6), M_q0m =
q0's own family (Gaussian, homoscedastic) with the true conditional mean at that information.

| group | site | Δ(q0) | Δ(O_q0m) | Δ(M_q0m) | form share | mean share | Δ(q1) | LN gauge: Δ(q0) → Δ(M_q0m) |
|---|---|---|---|---|---|---|---|---|
| 2L one-pool | a2-QT | 0.241 [0.225, 0.254] | 0.0012 | 0.012 [0.007, 0.015] | 1.00 | 0.95 | 0.061 | – |
| 2L one-pool | h2-QT | 0.162 [0.151, 0.179] | 0.0012 | 0.008 [0.007, 0.011] | 0.99 | 0.95 | 0.034 | 0.086 → 0.007 |
| 2L two-pool | a2-QT | 0.404 [0.346, 0.462] | 0.0010 | 0.008 [0.007, 0.009] | 1.00 | 0.98 | 0.064 | – |
| 2L two-pool | h2-QT | 0.259 [0.240, 0.278] | 0.0010 | 0.006 [0.005, 0.008] | 1.00 | 0.98 | 0.041 | 0.103 → 0.004 |
| 3L | a3-QT | 0.076 [0.060, 0.099] | 0.0014 | 0.007 [0.005, 0.009] | 0.98 | 0.91 | 0.019 | – |
| 3L | h3-QT | 0.071 [0.056, 0.096] | 0.0014 | 0.006 [0.004, 0.008] | 0.98 | 0.91 | 0.016 | 0.049 → 0.005 |
| plateau | a2-QT | 0.217 | 0.0021 | 0.022 | 0.99 | 0.90 | 0.155 | – |
| plateau | h2-QT | 0.200 | 0.0021 | 0.014 | 0.99 | 0.93 | 0.088 | 0.152 → 0.013 |

- **All of q0's gap at the readout cuts is form.**
  - The six features q0 reads carry what the head needs: the matched oracle loses 0.001–0.002 nats.
  - This holds although the features do not determine c (§2): the directions they leave free barely matter to G.
    O_c, which has all of c, loses 0.000–0.0003.
- **90–98% of q0's gap is its linear mean.**
  - q0's own family with the true conditional mean and matched homoscedastic noise loses 0.004–0.022 nats.
  - The Gaussian shape and off-manifold part, M_q0m − O_q0m, is 0.004–0.020 nats: 2–9% of Δ(q0).
  - The reviewer's LN gauge removes 17–56% of q0's gap (REVIEW-interpretation P1). The mean then accounts for 90–96%
    of what is left.
- **q1 (round 1's quadratic latents) gets 56–85% of the way.** A nonlinear mean in the same variables is what is
  needed.
- **The plateau net is the exception, and only Δ_rms shows it.**

  | | q0 | O_q0m | O_c | M_q0m |
  |---|---|---|---|---|
  | Δ_rms at a2-QT / h2-QT | 0.40 / 0.38 | 0.12 / 0.12 | 0.003 / 0.003 | 0.11 / 0.10 |

  - With q0's six features, real states still miss per cell by 0.12 nats, 31–33% of q0's Δ_rms; given all of c they
    miss by 0.003.
  - The signed oracle gap (0.002) hides this: per-cell errors of both signs cancel.
  - The plateau's readout is a different function of c (unordered reliabilities, §4) from the Bayes summaries q0 reads.
    For the plateau, a third of q0's final-cut error is missing information, and P9 fails there.

### 3.2 QQ tokens and full cuts: two regimes

Signed Δ in nats; seed means [range]. "Form A" = form first; "form B" = cross-source variables first (q0 → q0x → O_c).

| group | site | Δ(q0) | Δ(O_q0) | Δ(O_c) | Δ(q0x) | Δ(M_c) | Δ(O_cIDr) | form share A / B / Shapley | cross-source share A / Shapley |
|---|---|---|---|---|---|---|---|---|---|
| 2L two-pool | h1-QQ | 0.345 [0.307, 0.382] | −0.000 | 0.000 | 0.327 | 0.007 | 0.000 | 1.00 / 0.95 / 0.97 | 0.00 / 0.03 |
| 2L two-pool | h1-ALL | 0.315 [0.292, 0.338] | 0.001 | 0.000 | 0.299 | 0.008 | 0.000 | 1.00 / 0.95 / 0.97 | 0.00 / 0.03 |
| 2L one-pool | h1-QQ | 0.337 [0.312, 0.378] | 0.340 | 0.018 | 0.171 | 0.021 | 0.000 | −0.01 / 0.46 / 0.23 | 0.95 / 0.72 |
| 2L one-pool | h1-ALL | 0.412 [0.380, 0.464] | 0.355 | 0.026 | 0.259 | 0.031 | 0.001 | 0.14 / 0.57 / 0.35 | 0.79 / 0.58 |
| 3L (s10, s12–s14) | h1-QQ | 0.043–0.079 | ≤ 0.002 | ≤ 0.002 | 0.040–0.074 | 0.001–0.015 | ≤ 0.001 | 0.95–1.00 / 0.89–0.94 / 0.92–0.97 | ≤ 0.02 |
| 3L s11 | h1-QQ | 0.386 | 0.361 | 0.028 | 0.086 | 0.044 | 0.000 | 0.07 / 0.15 / 0.11 | 0.86 / 0.82 |
| 3L | h2-QQ | 0.342 [0.093, 0.551] | 0.300 | 0.033 | 0.134 | 0.035 | 0.025 | 0.12 / 0.32 / 0.22 | 0.80 / 0.70 |
| 3L | h2-ALL | 0.351 [0.157, 0.516] | 0.299 | 0.035 | 0.199 | 0.040 | 0.028 | 0.19 / 0.52 / 0.36 | 0.72 / 0.56 |
| plateau | h1-QQ | 0.254 | 0.229 | 0.001 | 0.098 | 0.001 | 0.000 | 0.10 / 0.38 / 0.24 | 0.90 / 0.76 |
| plateau | h1-ALL | 0.256 | 0.227 | 0.001 | 0.188 | 0.001 | 0.001 | 0.11 / 0.73 / 0.42 | 0.88 / 0.57 |

Plain findings:
- **Own-source regime.** These are the two-pool 2L nets and the 3L nets after block 1 (s10, s12, s13, s14).
  - The QQ state, as far as everything downstream is concerned, is a function of the own source's (Q_s, S_s).
  - The oracle given q0's information loses nothing at QQ (≤ 0.002), and neither does the oracle given only
    (Q_s, S_s) without β_s (O_own, ≤ 0.004).
  - Most of q0's gap is its linear mean: M_q0 is within 0.001–0.013 nats of the oracle, i.e. 2–6% of Δ(q0), except
    3L s14 (18% of a 0.07-nat gap).
  - A linear map does not get there even with every source's latents: q0x loses 0.04–0.36.
- **Cross-source regime.** QQ tokens carry the other sources' readings (one-pool mixer head; block 2 in every 3L net;
  layer 1 of the plateau net).
  - Given only q0's information, the true conditional is no better than q0: in the one-pool nets O_q0 ≈ q0, and in 2L
    s2 O_q0 is worse (0.408 vs 0.378).
  - Given c, the gap falls by 78–100%, and the true conditional mean with Gaussian noise (M_c) gets within
    0.000–0.018 nats of the oracle.
  - So the missing structure is variables **and** a nonlinear mean at the same time. Neither alone does it:
    - q0x (variables, linear mean) loses 0.04–0.36;
    - M_q0 (nonlinear mean, q0's variables) loses 0.08–0.52.
- **The plateau net is in the cross-source regime, but unbound.** Its QQ states depend on the other sources' (Q, S)
  (cross step 0.229), and not at all on which SS is the token's own (layer binding term −0.000, §4).
- **Round 1's structure (i) is confirmed with no fitted family.** The cross-source step at 2L h1-QQ is 0.28–0.40 nats
  in the one-pool seeds and 0.000 in the two-pool seeds (P11). In 3L the same step appears after block 2 in every seed
  (0.09–0.54), and after block 1 only in s11.

### 3.3 Order dependence (coordinator: no additive "shares" of RMS gaps)

- The decomposition is on signed Δ along fixed nested orders. Each order telescopes exactly, but **the attribution
  depends on the order**, and here it depends strongly.

  | site (cross-source regime) | form, form first | form, variables first | interaction (B − A) |
  |---|---|---|---|
  | 2L one-pool h1-QQ | −0.005 | 0.153 | +0.16 nats |
  | 3L h2-QQ (5 nets) | 0.04 | 0.11 | +0.07 |
  | plateau h1-ALL | 0.03 | 0.19 | +0.16 |

- The interaction is positive everywhere: the form gap shows up only once the variables are present. That is why
  round 1's linear refinements helped 40–85% and then stalled.
- The (cross-source, IDs) order check moves the IDs step by ≤ 0.016 nats (tables): IDs barely interact.
- Δ_rms is the closure metric and is reported per rung (`r2/tables.md`). Differences of Δ_rms are never presented as
  shares.

### 3.4 What c does not carry

| net, site | Δ(O_c) | Δ(O_cID): + all IDs | Δ(O_cIDr): + own residual vector, labels | Δ(J): one joint draw (c + IDs) |
|---|---|---|---|---|
| 2L s0 / s2 / s3, h1-QQ | 0.023 / 0.007 / 0.024 | 0.021 / 0.001 / 0.021 | 0.000 / 0.000 / 0.000 | 0.020 / 0.001 / 0.020 (partial site) |
| 3L s12, h1-ALL | 0.018 | 0.018 | 0.001 | 0.000 |
| 3L s11, h2-ALL | 0.082 | 0.079 | **0.068** | **0.000** |
| 3L s14, h2-ALL | 0.081 | 0.067 | **0.067** | **0.000** |
| 2L two-pool; 3L h1 (s10, s13); plateau | ≤ 0.001 | ≤ 0.001 | ≤ 0.001 | ≤ 0.001 |

- **One-pool QQ tokens depend on their own source's labelled residual vector** beyond S_s, by up to 0.024 nats
  (tilted pooling; B2's residual-direction share). In 3L s12, binding happens in block 2, so the LAB tokens' raw
  residuals matter at h1.
- **3L s11 and s14 after block 2 carry cross-token nuisance structure.** With c, all IDs and each token's own source
  data kept, independent per-token draws still lose 0.07 nats; one joint draw loses nothing. The downstream needs the
  tokens' residual-direction content to be mutually consistent.
  - A family with tokens independent given φ, which covers q0 and every rung in METHOD below q1f, has an irreducible
    gap here.
  - These are exactly METHOD's q1f shared factors.
- **IDs:** O_c → O_cID changes Δ by ≤ 0.015 nats everywhere (P15).
- **The token's own IDs matter only before binding** (O_lat − O_c, P16 fails): 0.000–0.006 at 2L h1-QQ, but 0.26 at
  3L s12 h1 and 0.02 at 3L s11 h1. After binding, downstream reads the QQ content without matching IDs. This is the
  untested reading in REVIEW-interpretation P6, now confirmed with no fitted family.

## 4. Binding

| net | B_JS = E JS_net / E JS_Bayes [95% CI] | layer term O_U − O_c at h1-QQ | at the last QT cut |
|---|---|---|---|
| 2L s0 / s1 / s2 / s3 / s4 | 0.906 / 1.021 / 0.924 / 0.923 / 0.993 (CIs ±0.003–0.004) | 0.163 / 0.264 / 0.225 / 0.166 / 0.264 | 0.275 / 0.303 / 0.281 / 0.277 / 0.296 |
| 3L s10 / s11 / s12 / s13 / s14 | 0.973 / 0.964 / 0.980 / 0.993 / 0.988 | 0.252 / 0.146 / **0.000** / 0.267 / 0.215 | 0.29–0.30 |
| plateau | **0.012** [0.012, 0.013] | **−0.000** | **−0.000** |

- **The whole-input measure is clean.**
  - The null control (unordered Bayes) gives 4e-16, and full Bayes gives JS = 0.124 nats.
  - The near-Bayes nets respond to the SS-to-source assignment as much as Bayes does (0.91–1.02).
  - The plateau net responds 1.2% as much. This reproduces astra's source-cycle intervention (symmetric-KL ratio 0.0)
    with a distribution-preserving, posterior-weighted measure.
- **The layer term agrees and localises binding.**
  - It is large wherever the net binds, and zero in the plateau net at every layer.
  - In 3L s12 it is zero after block 1 and 0.23 after block 2 (h2-QQ): binding happens in block 2 there.
  - The layer term stitches donor tokens into a real layer, so §5's caveat applies. The whole-input JS is the primary
    binding number.
- **q0 is not binding-deficient.** It reads its own source's latents bound to the token, so its gap has no binding
  share. The layer term measures what a binding-free description, reading only (Q, U), would lose.

## 5. Stitching and the joint manifold (astra's caution)

Whole-layer real-vs-stitched classifier, out-of-fold AUC (main cut: h1 for 2L and the plateau, h2 for 3L; 3,000
cells × 8 draws per class):

| group | site | q0 | O_q0 | O_c | O_cID | O_cIDr | J |
|---|---|---|---|---|---|---|---|
| 2L (5 nets) | QQ | 0.981–0.997 | 0.960–0.991 | 0.958–0.978 | 0.952–0.976 | 0.500 | 0.954–0.976 |
| 2L (5 nets) | ALL | 0.998–1.000 | 0.995–0.999 | 0.994–0.999 | 0.989–0.999 | 0.50–0.95 | 0.498–0.505 |
| 3L (5 nets) | h2-QQ | 0.993–0.999 | 0.969–0.999 | 0.926–0.999 | 0.932–0.986 | 0.50–0.97 | 0.930–0.978 |
| 3L (5 nets) | h2-ALL | 0.998–1.000 | 0.997–1.000 | 0.990–0.999 | 0.980–0.996 | 0.84–0.99 | 0.501–0.504 |
| plateau | QQ / ALL | 0.999 / 0.999 | 0.990 / 0.999 | **0.500** / 0.999 | 0.500 / 0.999 | 0.500 / 0.997 | 0.500 / 0.501 |

- **Astra is right that stitched layers leave the joint manifold.**
  - In the near-Bayes nets, every per-token oracle short of "c + all IDs + own source data" is detectable at AUC
    ≥ 0.92.
  - So is J at a partial site: QQ tokens from one donor, LAB tokens real.
  - The classifier works: J at a full cut, which is an exactly real layer, is at 0.50 in all 11 nets.
- **Detectability is not G-relevance.**
  - In the two-pool nets the stitched O_q0 and O_c layers are detectable at AUC 0.96–0.999, yet they cost
    0.000–0.001 nats.
  - So the network's downstream ignores the joint inconsistency there.
  - Where stitching does cost (3L s11 and s14 at h2, §3.4), it is because the downstream uses cross-token consistency.
- **Consequence for reading oracle gaps.** In the own-source regime the oracle's ≈ 0 is not an artefact. In the
  cross-source regime, O_q0's large gap could in principle include a reaction to inconsistency. Two facts argue
  against that:
  - O_c, which is equally stitched (AUC 0.96–0.98), loses only 0.007–0.08;
  - the size of the cross-source step is stable across the two orders (§3.3).

## 6. Readout control for localisation (estimator review fix 2; `r2loc.py`)

Fraction of each rung's own gap that remains when the named part is kept real and the rest is drawn from the rung
(knock-in; 3,000 fresh cells × 8). U_G = top-4 eigenvectors of E_p[∇G ∇Gᵀ] on real draws (no classifier, no q).
Components are written unscaled (the round-1 rescaling is dropped, REVIEW-interpretation P2). Ranges are over
all 11 nets (`r2/loc_<net>.json`).

| part kept real | site | q0: gap left | R_in: gap left | random 4-d (q0) |
|---|---|---|---|---|
| U_G (4 directions) | QQ (16 site-nets) | 0.004–0.047 | 0.003–0.027 | 0.10–0.91 |
| U_G (4 directions) | final QT (16) | 0.003–0.067 | 0.000–0.007 | 0.35–0.93 |
| MLP write | first post-block cut (11) | 0.003–0.017 | 0.000–0.003 | – |
| best head | first post-block cut, own-source nets: 2L s1, s4; 3L s10, s13 | 0.94–0.99 | 0.42–0.73 | – |
| best head | first post-block cut, own-source nets: 3L s12, s14 (h1 also carries own-residual information, §3.4) | 0.63–0.77 | 0.36–0.44 | – |
| the mixer head | first post-block cut, cross-source nets: 2L s0, s2, s3; 3L s11; plateau | 0.27–0.54 | 0.06–0.16 | – |

- **Both round-1 localisations are readout (P18 holds).**
  - Four ∇G directions remove 97–100% of R_in's gap at QQ and final query-token sites, a rung with no Bayes latents
    and a gap 1.5–50× q0's.
  - So does the MLP write.
  - So "4 directions carry the gap" and "the MLP write carries the gap" describe where downstream reads, and the fact
    that the rest of a block's write is Gaussian-predictable from any φ once the MLP output is real. They do not
    describe q0's missing structure.
- **What survives the control: the heads.**
  - In the own-source nets 2L s1, s4 and 3L s10, s13, keeping a head real removes ≤ 6% of q0's gap but 27–58% of
    R_in's: the heads compute the own-source latents that q0 already has.
  - In cross-source nets, keeping the mixer head real removes 46–73% of q0's gap: it carries the cross-source content
    that the oracle ladder finds missing (§3.2).
  - This is the one localisation that is specific to q0, and it agrees with the oracle and with REVIEW-interpretation
    P2's knock-out reading.
- **L(α):** no interior is reported. The endpoints are the direct means; the estimator review showed the interior moves
  with the classifier's training size.

## 7. Pre-registered predictions (PREREG amendment 4; outcomes from `r2/predictions.md`)

Generated tables: `r2/predictions.md`.

| | prediction (abridged) | outcome | numbers |
|---|---|---|---|
| C1 | V6: J at full cuts and O_c at final cuts give Δ within 2 SE and Δ_rms ≤ ε | Δ: **holds** 38/38; Δ_rms: **unresolvable** | max \|z\| 1.38; the null's Δ_rms upper bound 0.0006–0.005 > ε |
| C2 | joint classifier on J ≤ 0.52 | holds 11/11 | 0.498–0.505 |
| C3 | JS null and quadrature | holds 11/11 | ≤ 4e-16; grids agree to 3e-13 |
| C4 | β match | holds 11/11 | median 0.0031, 99th percentile 0.045 |
| P9 | final QT: Δ_rms(O_q0) ≤ 0.1·Δ_rms(q0) | **fails** 20/22 | near-Bayes 0.01–0.04; plateau 0.31–0.33 |
| P10 | form share ≥ 0.5 at read-heavy context sites in ≥ 80% | **fails** 4/22 | holds only in the two-pool seeds (1.00; its ≥ 0.8 part holds 4/4); cross-source sites −0.08 to 0.44 |
| P11 | cross-source step at 2L h1-QQ: ≥ 0.05 one-pool, ≤ 0.02 two-pool | **holds** | 0.28 / 0.40 / 0.28 vs −0.000 / −0.000; the reviewer's Δ_rms version also holds |
| P12 | layer binding at h1-QQ: ≥ 0.1 near-Bayes 2L, ≤ 0.02 plateau | **holds** | 0.16–0.26 vs −0.000 |
| P13 | B_JS ≥ 0.75 near-Bayes, ≤ 0.05 plateau | **holds** | 0.91–1.02 vs 0.012 |
| P14 | mean ≥ 0.6 of form at final QT and the main full cut | **holds** 29/29 | final QT 0.90–0.99 (one site-net excluded: form < 2 SE) |
| P15 | \|O_c − O_cID\| ≤ 0.02 everywhere | **holds** 60/60 | max 0.017 |
| P16 | own IDs (O_lat − O_c) ≥ 0.1 at 2L h1-QQ | **fails** | 0.001–0.006; IDs matter only before binding (3L s12 h1: 0.26) |
| P17 | AUC order q0 > O_q0 ≥ O_c > J at the full cut in ≥ 8/11 | holds 10/11 | ceiling: 0.990–1.000 for every per-token variant |
| P18 | readout control: U_G and MLP knock-in remove ≥ 90% / 80% of R_in's gap | **holds** | (a) 41/41 site-nets; (b) 11/11 nets (R_in left ≤ 0.003) |

- **The risky predictions split.**
  - Held: the motif (P11), binding in both senses (P12, P13), and the mean dominating form (P14).
  - Failed: "form dominates at QQ and full cuts" (P10, the reviewers' P1 reading, as a general claim); ID routing after
    block 1 (P16); and P9 in the plateau net.
- **P10's failure is the main correction to the reviews.** At query-question tokens in cross-source nets, q0's gap is
  mostly missing variables in the form-first order. Form dominates only where QQ tokens summarise their own source.

## 8. What this implies for the next tractable family

1. **Model the mean, not the shape.**
   - At every site, q0's family with the true conditional mean is within 0.000–0.020 nats (signed) of the on-manifold
     oracle at the same information. That is ≤ 9% of Δ(q0), except 18% at 3L s14 h1 where the gap is small.
   - A Gaussian with homoscedastic noise is good enough once the mean is right. Resampling bases, mixtures and flows
     are not needed for these sites.
   - This retires round 1's "off-manifold noise" reading, in line with the estimator review §2.1.
2. **The mean must be nonlinear, and it must see the right variables.** The variables, per site, are:
   - the own-source latents (Q_s, S_s) everywhere;
   - the other sources' (Q, S) at QQ tokens wherever attention mixes sources (one-pool nets after block 1, every 3L
     net after block 2, the plateau net after block 1);
   - the own source's residual vector, i.e. tilted pools, worth up to 0.024 nats at one-pool QQ;
   - the token's own IDs only before binding (3L s12 after block 1).

   Linear maps of exactly these variables (q0x) leave 0.04–0.36 nats.
3. **Proposed family for round 3:** h_t ~ N(μ_θ(φ_t), Σ_τ), with μ_θ a small MLP per token type on φ_t = the variables
   above. Score it on the direct Δ and Δ_rms with the oracles as floors. The floors are:
   - O_q0 (information-matched) when φ = q0's variables;
   - O_c or O_cIDr when φ adds those variables.

   A tier-iii surrogate of the writing block is the mechanistic version of the same thing.
4. **3L after block 2 in s11 and s14:** per-token independence costs 0.05–0.07 nats on its own. Add shared
   cross-token factors (METHOD q1f), or describe that layer through the pushforward of the block-1 description.
5. **Stop the naming loop on linear features.** Round 1's accepted q1 rungs remove 40–85% because they approximate the
   nonlinear mean. The ladder now says which variables are needed; more linear features cannot close the gap.
6. **Open:** description length. The oracle shows what information suffices, not that a short description exists.
   ε-level closure needs about 10× more cells to be resolvable at all (§2).

## 9. Adversarial self-check

1. **"On-manifold" holds per token, not per layer.**
   - Every donor state is a real state given its ψ, but a stitched layer is not a layer the network produces: AUC
     0.92–1.00 (§5).
   - So an oracle gap at a partial site, or with per-token draws, can include the network's reaction to an inconsistent
     layer as well as its use of a missing variable.
   - Bound: where ψ suffices, equally detectable stitching costs ≤ 0.002 nats. The one place where stitching itself
     costs a lot is 3L s11 and s14 after block 2 (0.07 nats per token, 0 with a joint draw), and there it is the
     finding.
2. **The form/variables split is order-dependent, strongly so in the cross-source regime.**
   - Form-first gives form −8% to 44% of Δ(q0) (median 10%); variables-first gives 14–82% (median 43%). The Shapley
     average is a convention.
   - Order-free statements:
     - at q0's information the true conditional is no better than q0;
     - at c-information the remaining gap is the mean (M_c ≈ O_c).
3. **G rewards concentration.**
   - The conditional mean without noise (M_*mean) beats the network itself at the final cuts (signed Δ down to
     −0.008), because it removes the network's nuisance-driven spread.
   - M_q0 beats O_q0 in the cross-source regime: hedging at the mean beats committing to a wrong cross-source value.
   - So "mean ≈ closes" is claimed only with noise at the matched within-ψ covariance (M_q0, M_c), never for M_*mean.
     A negative "shape" part means G prefers the Gaussian, not that the Gaussian has the right shape.
4. **M_q0 is only approximately "q0's family with the true mean".**
   - Its mean is an 8-draw average, so its error (covariance Σ/8) has the on-manifold shape.
   - Its noise is homoscedastic (pooled Σ_w).
   - The mean/shape split is therefore approximate at the 1/8 level. A heteroscedastic Gaussian was not tested.
5. **A pre-registered premise failed** (q0's six query-token features do not determine c).
   - It was repaired post hoc with a nearest-neighbour oracle (0.027 sd in feature space), which is itself an
     approximation.
   - In signed Δ the two versions agree to ≤ 0.002 nats everywhere, and in Δ_rms they agree in the near-Bayes nets
     (0.006–0.009).
   - In the plateau net they differ in Δ_rms (0.12 vs 0.003), so P9's plateau failure rests on the repair. The same
     matching gives 0.006–0.009 in the other ten nets, so matching noise cannot explain 0.12, but a finer pool would
     tighten it.
6. **ε-level closure is not resolvable.** The Δ_rms upper bound of an exact null is 0.0006–0.005 nats at n = 6,000,
   above ε for every near-Bayes net. "Loses ≤ 0.002" below means "Δ within 2 SE of 0 at the null's noise floor", not "≤ ε".
7. **Heavy tails.** Hill indices < 2 at many sites; within-seed CIs under-cover. The cross-seed ranges are the honest
   error bars, and every regime statement holds in every seed listed.
8. **Seeds and motifs.**
   - The 2L motif split rests on 3 one-pool vs 2 two-pool nets, the same 5 nets B2 and round 1 used, so it is agreement
     between methods, not replication.
   - The plateau net is n = 1 and was trained differently (cold start, KL 15× larger).
9. **The oracle is a diagnostic, not a description.** It uses the network as its own sampler. "O_c ≈ 0" says that c
   carries what downstream needs, not that a short description exists; description length is untouched this round.
10. **Binding at a layer is confounded with stitching.** The whole-input JS is stitching-free and is the primary number;
    the layer term is reported as localisation only.
11. **What could still be wrong.** The four reasons for a large gap (missing variables, nonlinear mean, wrong shape,
    off-manifold response) are separated here only up to the order effect (2), the concentration effect (3) and the
    stitching effect (1). The decisive test of the next family is not this decomposition: it is whether a fitted
    nonlinear-mean Gaussian reaches the oracle floors on fresh data.

## Files (`/home/agent/workspaces/mats/work/cm-interp/`)

- **Pre-registration:** `PREREG.md`, amendments 4 (design and predictions), 5 (M_c, after the smoke run) and 6 (the
  matched query-token oracle, forced).
- **Code:**
  - `lib/oracle.py`: exact samplers, β match, the per-token forward;
  - `tests/test_oracle.py`;
  - `r2run.py`: per net;
  - `r2qtpool.py` and `r2qt.py`: the matched query-token oracle;
  - `r2qtinfo.py`: the premise check;
  - `r2loc.py`: readout controls;
  - `r2summ.py`: tables and predictions.
- **Results:**
  - `r2/<net>.json` and `r2/<net>_D.npz`: per-draw G differences for every site and variant;
  - `r2/qt_<net>.*` and `r2/loc_<net>.json`;
  - `r2/qtinfo.json`;
  - `r2/tables.md`: every rung, per net, with SEs, Δ_rms CIs, Hill indices and trimmed means;
  - `r2/predictions.md`;
  - `r2/logs/`.
- **Runners:** `r2/run_all.sh` (r2run, 4 processes) and `r2/run_follow2.sh` (pool, r2qt, r2loc).
- **Scratch:** the 4,000,000-episode pool for r2qt is regenerated by `r2qtpool.py` (seed 75000, about 1 min).
- **Smoke run** at 300 cells, kept for the record: `r2/2L_s0_quick.*`.
