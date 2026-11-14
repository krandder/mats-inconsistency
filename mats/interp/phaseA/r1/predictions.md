# Pre-registered predictions, evaluated

- P1 q0 does not close: rms>eps at every site in 7/10 nets; full-cut Δ > KL_model in 10/10 -> FAILS
- P2 plateau plat_s3: rms(q0U) <= 1.2 rms(q0)+2SE at h1-QQ/h1-ALL/h2-QT: [True, True, True]; rms(q0U) at h1-ALL 0.3100 vs eps 0.0146 -> FAILS
- P2 near-Bayes 2L: rms(q0U) >= 2 rms(q0) at all three sites in 0/5 -> FAILS; per site ratios: h1-QQ 0.93,1.28,0.83,0.93,1.26; h1-ALL 1.11,1.30,0.97,1.14,1.34; h2-QT 1.41,1.41,1.57,1.52,1.30
- P3 3L smaller residual: last QT 2L 0.5259 vs 3L 0.2675; full cut 2L 0.6567 vs 3L 0.4503 -> HOLDS
- P4 motif: one-pool h1-QQ Δ [0.3225, 0.3815, 0.3138] vs 2x two-pool mean 0.7476: 0/3; naming picks cross in 3/3 -> FAILS
- P5 attenuation where AUC<0.6 (and Δ significant): bridge/direct<1 at 6/6 sites -> HOLDS
-    (descriptive) bridge/direct where AUC>=0.9: median 0.985, range 0.832-0.997, n=64
- P6 IDs: Δ(q0noID) >= 2Δ(q0) at h1-ALL in 0/5 2L seeds; ratios 1.00,0.99,1.00,0.99,1.02 -> FAILS
- P7 tier i max 0.0e+00; tier ii fails for 2L h1->h2 in 5/5 -> HOLDS
- P8 q1 accepted at 2L h1-ALL in 5/5; still > eps in 5/5 -> HOLDS
- Naming rule vs best add-one on B (sites where the best add-one cuts rms by >10%): same candidate 47/68; chosen achieves >= 80% of the best reduction 56/68
