codex
Reviewed read-only, checked at **2026-10-31 12:07 UTC**. Report SHA-256 prefix: `a4cf451b5d140e34`. **Verdict: major revision.**

1. **Major — Report §2, §4; summary items 1 and 4: missing regularity hypotheses.** The Lipschitz dimension lower bound needs a controlled encoding radius, `log(1+ργ)=o(log(1/ε))`, and sufficiently small exceptional encoding mass. A decoder’s Lipschitz constant alone is insufficient: an increasingly long scalar encoding range permits arbitrarily accurate quantization. Raw-weight decoding is **locally** Lipschitz, not globally budgeted and exact.  
   **Fix:** State the encoding domain, decoder norm/uniformity over backs, budget growth, and `n≥2`. For streaming, impose these on the actual cut state and composed final readout; regularity of the last update alone is insufficient.

2. **Major — Report §2 analytic row; FORMAL-ext Addendum 2: the upper bound’s radius restriction disappeared.** The multilinear construction and its `3^B max|f_b|` bound are correct on the **radius-2 polydisc**, corresponding to radius ratio `1/2`. They do not establish the displayed upper bound for arbitrary fixed radius ratio with the same amplitude threshold. Both front and back clipping are needed for a uniform amplitude bound.  
   **Fix:** Restrict this upper bound to ratio `1/2`, or make the amplitude threshold depend on the radius. Also handle the polylogarithmic factors at equality: the sharper interpolation bound `√5^B max|f_b|` validates the advertised `κ≥p log₂3` threshold at radius 2.

3. **Major — Report §2 “Cᵏ, upper bound”; FORMAL-ext Addendum 1: two different smoothness conclusions are conflated.** An inverse satisfying `|t−t′|≤C|y−y′|ᵏ` supplies a zero-derivative Whitney jet of class **Cᵏ⁻¹,¹**, not automatically Cᵏ. Cᵏ requires the corresponding little-o remainder.  
   **Fix:** Retain Addendum 1’s actual Cᵏ⁻¹,¹ conclusion. For Cᵏ, use Astra’s `(k+1)p` Cantor construction with inverse exponent **strictly greater than k**. That construction survives review.

4. **Major — Report §2 C3 and §8: exactness and ensemble quantifiers are overstated.** A curve onto a clipped weight ball gives exact decoding only on that ball. The ray mean works only for ray laws; clipped Whitney constructions give ensemble approximation, not exactness for every Gaussian-weight realization. Also, the genuine-ensemble `n(n+1)/2` chart bound does **not** transfer to ray fronts, whose variable-scale family has dimension n.  
   **Fix:** For globally exact scalar coding, explicitly use a continuous surjection `ℝ→ℝᵖ` with a measurable selector. Replace §8’s blanket “every net and every eps” and “or for ray fronts” statements with result-specific qualifications.

5. **Major — Report §3 frame naming; summary item 7: nonadaptivity is missing from the advertised atlas bound.** Theorem D is correct as formally stated. D1’s linear-rank consequence and D2’s `2^B binom(s+p,p)` bound require backward naming and output coefficients independent of front replies. Adaptive threshold decisions can create infinite separation rank. Similarly, the polynomial-interaction result requires the **final output**, as well as every message, to obey the polynomial/no-branching restriction.  
   **Fix:** Carry those restrictions into every summary statement. For arbitrary adaptive moment queries, retain only the information obstruction `e_K²`.

6. **Minor — Report §5: clarify what the Gaussian continuation proves.** At `t=−1`, the closure is **exact for the Gaussian**; the nonzero quantity is its difference from the analytically continued *physical-ray target* `qμ_t`. It is not Gaussian closure bias. Choose `q=wᵀv>0` explicitly, define `b_K` as restricted ensemble RMS error, and specify the extra `(K+1)`-cumulant trace for odd K.  
   **Fix:** Preserve the one-back-layer, exact-input qualification throughout.

   **Mathematical checks that passed:** the any-round `n log₂(1/ε)−O(1)` bit bound; the covariance chart and stronger one-way bit bound; cone-face moment recursion, Hilbert-basis determinacy and continuous decoding; fixed-budget floors under their actual bounded-domain hypotheses; and the polynomial rank lifting. The `α_t` family is realizable in the support of the K5 event for `c≥3`: use first-layer features `X₊, (−X)₊`, transmit them, then apply the positive bottleneck. Its variance stays positive on the real axis, so the finite closure is analytic there. The quantitative continuation argument supports the stated weak oracle floor. The streaming cut reduction itself is valid. Gaussian truncation supports §9’s cubature comparison, with truncation factors; it remains a sampling method.

7. **Major — Report §7; coverage_map rows K=1, covariance and cumulant propagation: coverage is internally inconsistent.** “Only dimension bounds apply, and its states satisfy them” is unjustified without uniform regularity assumptions. Specifically, K=1 stores `n+1` numbers, which is **less** than `n(n+1)/2` for `n≥3`. CNX22 D.3 is also misclassified as exact Gaussian **ReLU** covariance propagation: the fact sheet describes arithmetic-circuit product gates. WhestBench’s implementation was not verified.  
   **Fix:** Separate the variants and sources; mark regularity-bound applicability conditional, and distinguish satisfying a state-count threshold from achieving arbitrary accuracy.

8. **Minor — Coverage completeness and exclusions.** The map still lacks explicit treatment of CNX22’s arithmetic-circuit cumulant estimator (E1.4) and Xu’s three #SAT examples (E9d). “Sampling and hybrids: outside by construction” contradicts the map’s own sampled-state Lipschitz example. Enumeration does not inherently make an estimator discontinuous. “The only ARC estimator with a backward component” exceeds the fact sheet’s search coverage.  
   **Fix:** Add/group the missing entries; describe exclusions by estimand, access and regularity, and qualify the backward-pass claim as “among the sources inspected.”

9. **Major — NC2; [nl_readout.py:76](/home/agent/workspaces/mats/work/cut-rank-ext/scratch/nl_readout.py:76): the claimed bit count omits side information.** The norm code’s decoding interval depends on `s_ref=√(tr(m2)/n)`, a front-dependent real that is never transmitted or charged. Scaling a front preserves its quantized indices while changing the decoded vector through `s_ref`. Thus this is not an `nq+16`-bit protocol.  
   **Fix:** Use a globally specified norm codebook with overflow handling, or transmit and count the scale. Withdraw the “48 bits” communication claim and P6’s theorem interpretation pending that correction. P5 remains a quantization-error measurement conditional on the free scale.

10. **Major — NC1 ARC3; [nl_readout.py:131](/home/agent/workspaces/mats/work/cut-rank-ext/scratch/nl_readout.py:131) and [arc_unit_rerun.py:88](/home/agent/workspaces/mats/work/cut-rank-ext/scratch/arc_unit_rerun.py:88): incomplete oracle state.** Both supply only cumulants 1–3. Wu’s basic K=3 algorithm also requires the scalar harmonic component of cumulant 4, confirmed by paper pp. 47–48 and `get_r_x`. `coerce_input` does not manufacture it.  
    **Fix:** Label these runs as an incomplete-input variant, or rerun with the required trace. Their instability cannot currently be attributed to the correctly supplied exact-input ARC3 estimator.

11. **Major — Report §6 and summary item 9: “exact cut moments/cumulants” is false for the measurements.** The inputs are estimates from **8,192 sampled directions**. Only radial integration is exact. Nonlinear closure can amplify moment-estimation noise, particularly at K=3.  
    **Fix:** Say “sample-estimated cut moments with exact radial integration.” Keep these experiments distinct from the exact-input theorem; demonstrate convergence in input sample count before interpreting measured error as oracle closure bias.

12. **Major — Report §6 SVD interpretation and summary item 9: empirical estimates are presented as universal bounds.** `tails_in` uses eigenvalue tails of a noisy cross-Gram matrix. Its optimistic bias does not make each realized number a certified lower bound for every population linear protocol. Nor is the normalized `Err/Vbar` a per-network relative-variance guarantee.  
    **Fix:** Call these optimistic empirical tail estimates under the registered normalized loss. Qualify “best linear leaves 500 times as much”; certification needs uncertainty control or further independent evaluation.

    Recomputing all four NC1 configurations from NPZ files reproduced the saved JSON scores:

    | Configuration | MF | ARC2 unit-scaled | LIN2 | SVD(n+1) estimate |
    |---|---:|---:|---:|---:|
    | 8,4,4 | 0.0462240 | 0.0812047 | 0.0225474 | 0.0512727 |
    | 8,8,8 | 0.0309813 | 0.948923 | 0.0731002 | 0.0868475 |
    | 16,8,8 | 0.0588775 | 0.0635967 | 0.211394 | 0.119728 |
    | 8,64,8 | 0.000175125 | 0.708177 | 0.107378 | 0.0905053 |

    NC2 reproduces slope **−2.115427**, using q=1–5, nominal threshold **48**, and `log₂r_in(0.1)=4.807355`. The numerical scorecard outcomes reproduce, subject to finding 9.

13. **Minor — Preregistration and amendment reporting.** Both hashes verify: the original pre-amendment text gives `d8b1b493…`; the current amended file gives `8d5311a7…`. However, “pilot … never scored” is contradicted by `pilot_n8_c2_d2_scores.json`, containing full primary and raw scores. The report promises both ARC versions but omits the original merged ARC2 error, **1.49914×10⁹**, versus **0.708177** after rescaling.  
    **Fix:** Disclose pilot scoring without implying unestablished tuning; show original and amended results together. The post hoc diagnostics are appropriately identified in Amendment 1, but summary item 9’s blanket “pre-registered” label should distinguish them.

14. **Minor — NC3 and numeric descriptions.** Recomputed resolved-direction counts are `[8,7,7]`, `[3,5,6]`, `[11,13,16]`, `[12,12,8]`, agreeing with the report. But nonzero spectra span **3.38–8.15 decades**, not universally three to four. LIN2 at width 16 has **153**, not 154, features. The report omits registered rejection counts **2,6,0,2** and the merged configuration’s **211 surviving test fronts**.  
    **Fix:** Correct these descriptions and call machine-scale singular values numerical zeros unless an exact symmetry is certified. The Jacobian finite-difference check independently reproduced **6.30×10⁻¹⁰** relative discrepancy.

15. **Major — Summary fidelity.** Beyond the issues above, “no summary-size bound can cover them” is too broad: the report itself supplies bit/dimension restrictions and an oracle state-size lower bound. “No symmetry beyond the known ones shows up on most fronts” invites an identifiability conclusion that the unresolved spectra cannot support. Bit bounds also lose their additive constants and lower-order upper-bound terms.  
    **Fix:** Say unrestricted communication defeats the proposed precision-power separation; retain the **basic, one-step, exact-input** scope of the ARC theorem; describe NC3 as inconclusive about remaining directions; restore asymptotic qualifications.

**Verdict: major revision.** The central communication-versus-computation conclusion survives. The three priority fixes are **(1)** restore theorem domains, budgets and adaptivity restrictions, **(2)** repair NC2’s uncounted scale, and **(3)** correct ARC3’s supplied state and all “exact-input” experimental claims.
