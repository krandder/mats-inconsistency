codex
**Much improved, but not yet accurate enough to sign off.** Read-only checks against the revised report, logs and saved arrays completed 29 October 2026, ~20:18 UTC.

“FIXED” includes withdrawing an unsupported claim; it does not imply the missing experiment was performed.

| GPT-6 finding | Status | Remaining issue |
|---|---|---|
| 1. Proportional-depth theorem | FIXED | Joint-limit application explicitly empirical. |
| 2. Radial row | PARTLY FIXED | Growth corrected, but C2 still says “PROVEN UNDER” assumptions that do not control the full Jacobian’s top singular value. |
| 3. Wrong family/chart predictors | PARTLY FIXED | Mismatch disclosed; actual polar-family tails remain uncomputed. Optimal-block forecasts still inherit it. |
| 4. Sampling lower bound | PARTLY FIXED | §10.3 still says a positive result “would need” preserved closed-form structure. Unsupported necessity. |
| 5. Conditioned-chain exponents | PARTLY FIXED | Common angular truncation fixed; radial expectation needs the same explicit conditioning, and the gap index needs clarification. |
| 6. Fixed-radius sharpness | FIXED | Replaced by local sharpness. |
| 7. Applying chi-square bound to empirical families | FIXED | Quantitative applicability explicitly withdrawn. |
| 8. Wrong test of approximation A1 | FIXED | Linearisation and independence steps distinguished; A1 labelled untested. |
| 9. Noise-integrated versus pathwise Jacobian | FIXED | Operators distinguished. |
| 10. Fit strength and uncertainty | PARTLY FIXED | Pooled net-level fits added; §6 still calls training residual scatter predictive accuracy. No held-out validation or net bootstrap. |
| 11. Mechanism inferred from slope | FIXED | Mechanism claim and erroneous cancellation ratio removed. |
| 12. Class-count lower bound | PARTLY FIXED | Coverage disclosed, but new “the two bracket it” claim is unjustified; no confidence bounds or training-size convergence. |
| 13. Single-path normalization | PARTLY FIXED | Net 113 corrected; net 114 still uses mean-of-ratios under a trace-weighted heading. |
| 14. Huge class-count extrapolation | FIXED | Explicitly illustrative; coherence fractions corrected. |
| 15. Composition requirements | PARTLY FIXED | §10.1 says forecast, but summary item 4 and §10.3 still state necessary dimensions. |
| 16. Width scaling | PARTLY FIXED | Asymptotic claim softened; interpolated counts still labelled “MEASURED,” and “saturates” exceeds three-width evidence. |
| 17. QMC exclusion | FIXED | Left open. |
| 18. Single-path qualifications | PARTLY FIXED | Main theorem repaired; summary still omits survival/initial-active-fraction conditions and overstates the full-spectrum consequence. |
| 19. Equality and spectral gaps | FIXED | Common-maximizer/gap qualifications added. |
| 20. Numerical precision | FIXED | Sample inactivity, horizon variation and VJP discrepancy corrected. |
| 21. Reproducibility/counts | FIXED | Missing logs supplied; dense-spectrum thresholds reproduce. |

| Gemini finding | Status | Remaining issue |
|---|---|---|
| 1. C1 extinction/indexing | PARTLY FIXED | Same residual notation issues as GPT-6 #5. |
| 2. C2 normalization/limits | PARTLY FIXED | Radial edge control remains unproved despite theorem label. |
| 3. Theorem A sharpness | FIXED | Local formulation also avoids Gemini’s incorrect fixed-radius “supremum” remedy. |
| 4. Theorem B assumptions | FIXED | Specified residual swap is centered; pseudoinverses and approximation status added. |
| 5. PR denominator | FIXED | Single-path denominator is now \(2+2L'\). |
| 6. Theorem C qualifications | PARTLY FIXED | Main statement repaired; summary qualifications remain incomplete. |
| 7. Theorem D equality | FIXED | Common maximizer stated. |
| 8. Edge law | FIXED | Already correct; retained. |

Independent numerical spot-checks:

| Claim | Recomputed result | Assessment |
|---|---|---|
| Cut-32 pooled \(R^2\), scatter | 0.768496; ×1.56867 | Matches 0.77, ×1.6 **training RMS**. |
| Cut-48 pooled \(R^2\) | 0.511688 / 0.592621 | Matches, but uses **21 / 20 points**, respectively; §6 says 21 for both. |
| Optimal counts, targets \(10^{-5}/2×10^{-6}\) | 28/18; 57/38 | Matches. |
| PC counts, same targets | 51/33; 98/71 | Matches. |
| Width counts 44/43/42 | 43.53 / 43.34 / 41.97 | Reproduces **log-linear interpolation**, not measured threshold crossings. |
| Class gain at \(k=24\) | Best cross-fitted: 13.65% / 17.40%; net-113 in-sample: 29.87% | “14–30%” mixes evaluation methods. |
| Net-113 trace-weighted tail, \(k=32\) | \(7.10225×10^{-6}\) | Matches. Net-114’s \(5.73878×10^{-6}\) is **not trace-weighted**. |
| Leave-one-point-out slope range | 0.96124–1.19130 | §6’s 1.00–1.19 is too narrow. |

The main remaining overclaims are concentrated in these passages:

- [§7](/home/agent/workspaces/mats/outbox/lyapunov-dimension-2026-10-29.md:336): in-sample and cross-fitted estimates have different biases; their realized values do **not** guarantee a population bracket. Consequently, the summary cannot present 30% as a demonstrated generalization gain.
- [§9](/home/agent/workspaces/mats/outbox/lyapunov-dimension-2026-10-29.md:423): 44/43/42 are association-based interpolations. The width-128 count near 66 at \(2×10^{-6}\) additionally **extrapolates beyond the saved \(k=64\) ladder**.
- [Summary/§10](/home/agent/workspaces/mats/outbox/lyapunov-dimension-2026-10-29.md:471): “each layer needs” and “a positive result would need” remain stronger than the tested families and forecasts establish.
- Summary item 1’s **23** comes from finite-depth Fuss–Catalan, approximately **22.79**; its displayed asymptotic expression \(nG/(2L')\) gives **25.34**. Identify which formula supplies the number.

**Is the summary for Kelvin now accurate? No: its empirical core is sound, but the class-gain statement and necessary-dimension language still overclaim.**
tokens used
98,591
**Much improved, but not yet accurate enough to sign off.** Read-only checks against the revised report, logs and saved arrays completed 29 October 2026, ~20:18 UTC.

“FIXED” includes withdrawing an unsupported claim; it does not imply the missing experiment was performed.

| GPT-6 finding | Status | Remaining issue |
|---|---|---|
| 1. Proportional-depth theorem | FIXED | Joint-limit application explicitly empirical. |
| 2. Radial row | PARTLY FIXED | Growth corrected, but C2 still says “PROVEN UNDER” assumptions that do not control the full Jacobian’s top singular value. |
| 3. Wrong family/chart predictors | PARTLY FIXED | Mismatch disclosed; actual polar-family tails remain uncomputed. Optimal-block forecasts still inherit it. |
| 4. Sampling lower bound | PARTLY FIXED | §10.3 still says a positive result “would need” preserved closed-form structure. Unsupported necessity. |
| 5. Conditioned-chain exponents | PARTLY FIXED | Common angular truncation fixed; radial expectation needs the same explicit conditioning, and the gap index needs clarification. |
| 6. Fixed-radius sharpness | FIXED | Replaced by local sharpness. |
| 7. Applying chi-square bound to empirical families | FIXED | Quantitative applicability explicitly withdrawn. |
| 8. Wrong test of approximation A1 | FIXED | Linearisation and independence steps distinguished; A1 labelled untested. |
| 9. Noise-integrated versus pathwise Jacobian | FIXED | Operators distinguished. |
| 10. Fit strength and uncertainty | PARTLY FIXED | Pooled net-level fits added; §6 still calls training residual scatter predictive accuracy. No held-out validation or net bootstrap. |
| 11. Mechanism inferred from slope | FIXED | Mechanism claim and erroneous cancellation ratio removed. |
| 12. Class-count lower bound | PARTLY FIXED | Coverage disclosed, but new “the two bracket it” claim is unjustified; no confidence bounds or training-size convergence. |
| 13. Single-path normalization | PARTLY FIXED | Net 113 corrected; net 114 still uses mean-of-ratios under a trace-weighted heading. |
| 14. Huge class-count extrapolation | FIXED | Explicitly illustrative; coherence fractions corrected. |
| 15. Composition requirements | PARTLY FIXED | §10.1 says forecast, but summary item 4 and §10.3 still state necessary dimensions. |
| 16. Width scaling | PARTLY FIXED | Asymptotic claim softened; interpolated counts still labelled “MEASURED,” and “saturates” exceeds three-width evidence. |
| 17. QMC exclusion | FIXED | Left open. |
| 18. Single-path qualifications | PARTLY FIXED | Main theorem repaired; summary still omits survival/initial-active-fraction conditions and overstates the full-spectrum consequence. |
| 19. Equality and spectral gaps | FIXED | Common-maximizer/gap qualifications added. |
| 20. Numerical precision | FIXED | Sample inactivity, horizon variation and VJP discrepancy corrected. |
| 21. Reproducibility/counts | FIXED | Missing logs supplied; dense-spectrum thresholds reproduce. |

| Gemini finding | Status | Remaining issue |
|---|---|---|
| 1. C1 extinction/indexing | PARTLY FIXED | Same residual notation issues as GPT-6 #5. |
| 2. C2 normalization/limits | PARTLY FIXED | Radial edge control remains unproved despite theorem label. |
| 3. Theorem A sharpness | FIXED | Local formulation also avoids Gemini’s incorrect fixed-radius “supremum” remedy. |
| 4. Theorem B assumptions | FIXED | Specified residual swap is centered; pseudoinverses and approximation status added. |
| 5. PR denominator | FIXED | Single-path denominator is now \(2+2L'\). |
| 6. Theorem C qualifications | PARTLY FIXED | Main statement repaired; summary qualifications remain incomplete. |
| 7. Theorem D equality | FIXED | Common maximizer stated. |
| 8. Edge law | FIXED | Already correct; retained. |

Independent numerical spot-checks:

| Claim | Recomputed result | Assessment |
|---|---|---|
| Cut-32 pooled \(R^2\), scatter | 0.768496; ×1.56867 | Matches 0.77, ×1.6 **training RMS**. |
| Cut-48 pooled \(R^2\) | 0.511688 / 0.592621 | Matches, but uses **21 / 20 points**, respectively; §6 says 21 for both. |
| Optimal counts, targets \(10^{-5}/2×10^{-6}\) | 28/18; 57/38 | Matches. |
| PC counts, same targets | 51/33; 98/71 | Matches. |
| Width counts 44/43/42 | 43.53 / 43.34 / 41.97 | Reproduces **log-linear interpolation**, not measured threshold crossings. |
| Class gain at \(k=24\) | Best cross-fitted: 13.65% / 17.40%; net-113 in-sample: 29.87% | “14–30%” mixes evaluation methods. |
| Net-113 trace-weighted tail, \(k=32\) | \(7.10225×10^{-6}\) | Matches. Net-114’s \(5.73878×10^{-6}\) is **not trace-weighted**. |
| Leave-one-point-out slope range | 0.96124–1.19130 | §6’s 1.00–1.19 is too narrow. |

The main remaining overclaims are concentrated in these passages:

- [§7](/home/agent/workspaces/mats/outbox/lyapunov-dimension-2026-10-29.md:336): in-sample and cross-fitted estimates have different biases; their realized values do **not** guarantee a population bracket. Consequently, the summary cannot present 30% as a demonstrated generalization gain.
- [§9](/home/agent/workspaces/mats/outbox/lyapunov-dimension-2026-10-29.md:423): 44/43/42 are association-based interpolations. The width-128 count near 66 at \(2×10^{-6}\) additionally **extrapolates beyond the saved \(k=64\) ladder**.
- [Summary/§10](/home/agent/workspaces/mats/outbox/lyapunov-dimension-2026-10-29.md:471): “each layer needs” and “a positive result would need” remain stronger than the tested families and forecasts establish.
- Summary item 1’s **23** comes from finite-depth Fuss–Catalan, approximately **22.79**; its displayed asymptotic expression \(nG/(2L')\) gives **25.34**. Identify which formula supplies the number.

**Is the summary for Kelvin now accurate? No: its empirical core is sound, but the class-gain statement and necessary-dimension language still overclaim.**
