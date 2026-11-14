codex
**Verdict: minor revision.** Core repairs and reruns hold; remaining changes concern hypotheses, numerical interpretation and inconsistent wording. Read-only check at **2026-10-31 12:29 UTC**, [report](/home/agent/workspaces/mats/outbox/cut-rank-ext-2026-10-31.md) SHA prefix `cff2ab548678`.

| Finding | Status | Evidence / remaining issue |
|---|---|---|
| 1. Regularity hypotheses | PARTLY | §2 L128–129 and §4 restore radius/composition conditions; specify the decoder norm, uniformity over backs, exceptional-mass threshold and Cᵏ encoding domain. §1’s fixed-budget claim remains unqualified. |
| 2. Analytic upper bound | ADDRESSED | §2 L125, L145–148 restrict to radius ratio ½, clip both halves and use √5ᴮ. |
| 3. Cᵏ versus Cᵏ⁻¹,¹ | ADDRESSED | §2 L150–153 distinguishes the constructions and requires inverse exponent >k. |
| 4. Exactness/quantifiers | ADDRESSED | §2 L165–166 supplies the global surjection; §8 L359–373 separates ensemble, ray and clipped results. |
| 5. Nonadaptivity | ADDRESSED | Summary L37–39 includes polynomial final output; §3 L182–193 states naming/coefficient restrictions. |
| 6. Gaussian continuation | ADDRESSED | §5 L224–230 distinguishes Gaussian exactness from continued-target discrepancy, supplies odd-order trace and defines restricted RMS. |
| 7. Coverage consistency | PARTLY | §7 L320–322 fixes variants and conditional bounds; linked `coverage_map.md` L66–69 still says “only the dimension bounds apply, and its states satisfy them.” |
| 8. Coverage/exclusions | PARTLY | §7 L327–335 adds missing sources and qualifications; §9 L383 still incorrectly excludes all adaptive case splitting from every regularity class. |
| 9. NC2 free scale | ADDRESSED | §6 L289–292 discloses the defect; global-code rerun reproduces the threshold without side information. |
| 10. ARC3 missing trace | ADDRESSED | §6 L279–282 labels old runs incomplete and reports fourth-cumulant reruns. |
| 11. Estimated inputs | PARTLY | §6 L254–257, L272 and P3 now correctly say estimated; convergence interpretation at L276–278 is wrong. |
| 12. Empirical SVD bounds | PARTLY | §6 L283–285 correctly qualifies estimates; Summary L77 retains “per-net variance,” and L292 calls empirical `log₂r_in` a lower bound. |
| 13. Preregistration | ADDRESSED | §6 L247–257 discloses pilot scoring/amendments; L274–275 reports original versus rescaled ARC2. |
| 14. Numerical descriptions | ADDRESSED | §6 L268, L286, L297–301 fixes 211 survivors, 153 features, spectral range and rejection counts. |
| 15. Summary fidelity | PARTLY | Most scope/asymptotic qualifications restored; Summary L71–77 still conflates bit-bound regularity requirements and per-network loss. |

**Recomputed numbers**

- **NC2, merged:** slope **−2.11537752**, fitted over q=1–5; q=3 gives **0.01117299 Vbar**, q=4 gives **0.00285422 Vbar**. Thus the first passing allocation is **8×4+16=48 bits**; zero overflows.
- **ARC3:** reported ranges reproduce:

| Configuration | Median error / Vbar | Pairs >100 Vbar |
|---|---:|---:|
| (8,4,4) | 9.47037 | 32.2266% |
| (8,8,8) | 121181.755 | 81.7749% |
| (8,64,8) | 7738936.240 | 87.6037% |

- **Convergence:** §6 L277’s “errors … move by at most 0.002” confuses prediction-difference MSE with change in error:

| Configuration / estimator | Error: 8k → 64k | Error change | Prediction-difference MSE / Vbar |
|---|---:|---:|---:|
| (8,8,8), ARC2 | .908693 → .919514 | .010820 | .000356 |
| (8,8,8), K2X | .775508 → .784012 | .008504 | .000324 |
| (8,64,8), ARC2 | .756491 → .771726 | .015235 | .001540 |
| (8,64,8), K2X | .737014 → .751051 | .014037 | .001492 |

The merged subset has **53 surviving fronts**, not 64. These results support stability across these sample counts, not exclusion of all moment-estimation effects.

- **αₜ biases, rerun:** K=1 **3.264–49.168%**; K=2 **3.343–18.932%**; K=3 **1.257–59.014%**; K=4 **0.875–3.838%**. §5’s integer-rounded ranges are correct.

**Required before acceptance:** correct the convergence paragraph; finish the six partial items; synchronize the coverage map’s analytic, polynomial-interaction and streaming hypotheses. Change “one amendment” at L14 to two, remove `REVIEW2PLACEHOLDER` at L458, and qualify “All findings were applied.” No additional reruns are needed for these corrections.
