### Predictive mean vs the Bayes mean (MSE; separates deduplication from variance calibration)

Duplicate excess = MSE on duplicate episodes with all-singleton partitions minus MSE on all-singleton episodes without duplicates. Count-all (a) has excess ≈ 0.018; Bayes has 0. Fraction = model excess / (a) excess.

| depth | keys | n seeds | conv | MSE singletons, no dup | MSE dup | duplicate excess | fraction of (a) | MSE shared_any (a: 0.126, b: 0.116) | untrained MSE shared_any |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 3 | 3/3 | 0.027 ± 0.003 | 0.035 ± 0.002 | 0.008 ± 0.001 | 0.45 | 0.030 ± 0.000 | 0.85 |
| 1 | 1 | 3 | 2/3 | 0.008 ± 0.001 | 0.016 ± 0.000 | 0.008 ± 0.001 | 0.44 | 0.013 ± 0.001 | 0.83 |
| 2 | 0 | 3 | 1/3 | 0.007 ± 0.002 | 0.019 ± 0.002 | 0.012 ± 0.000 | 0.64 | 0.020 ± 0.002 | 0.96 |
| 2 | 1 | 3 | 2/3 | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.19 | 0.005 ± 0.000 | 1.44 |
| 4 | 0 | 3 | 0/3 | 0.005 ± 0.001 | 0.016 ± 0.001 | 0.011 ± 0.000 | 0.60 | 0.017 ± 0.000 | 0.93 |
| 4 | 1 | 3 | 0/3 | 0.005 ± 0.001 | 0.011 ± 0.002 | 0.005 ± 0.001 | 0.30 | 0.009 ± 0.002 | 1.02 |
| 8 | 0 | 2 | 0/2 | 0.024 ± 0.005 | 0.032 ± 0.004 | 0.008 ± 0.001 | 0.46 | 0.026 ± 0.000 | 0.94 |

### Probes (retrained ridge per layer) and attention, trained vs untrained (same init seed); mean over seeds

| depth | keys | n | group-size R² by layer, trained | untrained | query → Bayes-mean R², trained | untrained | corr(query attn per source, GLS weight) by layer, trained | untrained |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 3 | -0.00, 0.23 | -0.00, -0.01 | -0.03, 0.95 | -0.03, 0.16 | 0.20 | 0.00 |
| 1 | 1 | 3 | -0.02, 0.58 | -0.02, 0.04 | -0.03, 0.98 | -0.03, 0.13 | 0.35 | 0.03 |
| 2 | 0 | 3 | -0.00, 0.24, 0.61 | -0.00, -0.01, -0.01 | -0.03, 0.93, 0.97 | -0.03, 0.16, 0.17 | 0.09, 0.24 | 0.00, -0.02 |
| 2 | 1 | 3 | -0.02, 0.75, 0.83 | -0.02, 0.04, 0.12 | -0.03, 0.93, 0.99 | -0.03, 0.13, 0.13 | 0.21, 0.86 | 0.03, 0.01 |
| 4 | 0 | 3 | -0.00, 0.23, 0.58, 0.67, 0.68 | -0.00, -0.01, -0.01, -0.02, -0.02 | -0.03, 0.91, 0.96, 0.97, 0.98 | -0.03, 0.16, 0.17, 0.19, 0.19 | 0.08, 0.12, 0.17, 0.31 | 0.00, -0.02, -0.01, 0.02 |
| 4 | 1 | 3 | -0.02, 0.63, 0.76, 0.77, 0.77 | -0.02, 0.04, 0.12, 0.16, 0.19 | -0.03, 0.90, 0.96, 0.99, 0.99 | -0.03, 0.13, 0.13, 0.12, 0.14 | 0.22, 0.53, 0.67, 0.40 | 0.03, 0.01, -0.00, 0.03 |
| 8 | 0 | 2 | -0.00, 0.09, 0.26, 0.39, 0.43, 0.43, 0.44, 0.44, 0.45 | -0.00, -0.02, -0.02, -0.02, -0.02, -0.03, -0.03, -0.03, -0.03 | -0.03, 0.82, 0.93, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96 | -0.03, 0.17, 0.17, 0.18, 0.18, 0.19, 0.18, 0.18, 0.18 | 0.09, 0.05, 0.17, 0.11, 0.18, 0.17, 0.14, 0.11 | -0.01, -0.03, -0.01, 0.03, 0.02, 0.01, -0.00, 0.00 |

### Copy sweep: one source's row copied to m rows (m = 1..8; training only ever had m ≤ 2)

Pull = OLS slope of (μ_m − μ_1) on (o_dup − μ_1). True duplicates (Bayes): 0 for every m.

| model | n | m=2 | m=3 | m=4 | m=6 | m=8 |
|---|---|---|---|---|---|---|
| Bayes, true duplicates | | 0 | 0 | 0 | 0 | 0 |
| GLS + independent copies | | 0.176 | 0.296 | 0.384 | 0.506 | 0.587 |
| count-all (a) | | 0.121 | 0.216 | 0.293 | 0.408 | 0.491 |
| L1 keys 0 | 3 | 0.049 | 0.083 | 0.106 | 0.134 | 0.150 |  (untrained m=8: 0.003)
| L1 keys 1 | 3 | 0.040 | 0.062 | 0.075 | 0.090 | 0.096 |  (untrained m=8: 0.005)
| L2 keys 0 | 3 | 0.048 | 0.076 | 0.087 | 0.081 | 0.066 |  (untrained m=8: 0.003)
| L2 keys 1 | 3 | 0.030 | 0.048 | 0.063 | 0.084 | 0.101 |  (untrained m=8: 0.021)
| L4 keys 0 | 3 | 0.046 | 0.072 | 0.080 | 0.065 | 0.043 |  (untrained m=8: 0.011)
| L4 keys 1 | 3 | 0.034 | 0.054 | 0.067 | 0.085 | 0.097 |  (untrained m=8: 0.008)
| L8 keys 0 | 2 | 0.044 | 0.073 | 0.093 | 0.116 | 0.128 |  (untrained m=8: 0.004)

Log-precision shift (log σ_1 − log σ_m; positive = more confident). Bayes duplicates 0; independent copies m=2: 0.104, m=4: 0.260, m=8: 0.473.

| model | m=2 | m=4 | m=8 |
|---|---|---|---|
| L1 keys 0 | -0.003 | -0.019 | -0.048 |
| L1 keys 1 | -0.007 | -0.030 | -0.073 |
| L2 keys 0 | -0.009 | -0.050 | -0.118 |
| L2 keys 1 | -0.005 | -0.023 | -0.059 |
| L4 keys 0 | -0.010 | -0.053 | -0.122 |
| L4 keys 1 | -0.004 | -0.018 | -0.042 |
| L8 keys 0 | -0.006 | -0.026 | -0.060 |

### Query attention mass on the copies' query-question tokens (sum over copies), last layer

Perfect deduplication by readout attention: total constant in m (per-copy mass ∝ 1/m). Ratio = total(m=2)/total(m=1); per-head ratios in brackets.

| model | total m=1 | m=2 | m=4 | m=8 | ratio m2/m1 (heads) | untrained ratio | min head ratio over all layers |
|---|---|---|---|---|---|---|---|
| L1 keys 0 | 0.119 | 0.179 | 0.253 | 0.334 | 1.50 (1.34, 1.70, 1.45, 1.59) | 1.72 | 1.34 |
| L1 keys 1 | 0.151 | 0.230 | 0.329 | 0.437 | 1.52 (1.45, 1.54, 1.53, 1.60) | 1.71 | 1.45 |
| L2 keys 0 | 0.114 | 0.165 | 0.228 | 0.296 | 1.45 (1.48, 1.36, 1.74, 1.35) | 1.72 | 1.35 |
| L2 keys 1 | 0.124 | 0.170 | 0.226 | 0.288 | 1.37 (1.41, 1.50, 1.15, 1.25) | 1.72 | 1.15 |
| L4 keys 0 | 0.097 | 0.156 | 0.227 | 0.287 | 1.60 (1.65, 1.56, 1.54, 1.63) | 1.72 | 1.25 |
| L4 keys 1 | 0.089 | 0.126 | 0.172 | 0.223 | 1.42 (1.45, 1.49, 1.34, 1.43) | 1.72 | 1.20 |
| L8 keys 0 | 0.042 | 0.069 | 0.103 | 0.139 | 1.65 (1.64, 1.70, 1.68, 1.53) | 1.72 | 1.21 |

### Is the copy count linearly decodable from the copies? (retrained ridge, R² for log m, by layer; 0 = input)

| model | trained, copies' states | untrained | shuffled target | trained, within-episode h_m − h_1 | untrained within |
|---|---|---|---|---|---|
| L1 keys 0 | -0.00, 0.03 | -0.00, -0.00 | -0.00, -0.00 | 0.00, 0.15 | 0.00, 0.08 |
| L1 keys 1 | -0.00, 0.02 | -0.00, -0.01 | -0.00, -0.01 | 0.00, 0.09 | 0.00, 0.07 |
| L2 keys 0 | -0.00, 0.03, 0.15 | -0.00, -0.00, 0.01 | -0.00, -0.01, -0.01 | 0.00, 0.24, 0.42 | 0.00, 0.08, 0.13 |
| L2 keys 1 | -0.00, 0.20, 0.27 | -0.00, -0.01, -0.02 | -0.00, -0.01, -0.01 | 0.00, 0.10, 0.26 | 0.00, 0.07, 0.04 |
| L4 keys 0 | -0.00, 0.01, 0.16, 0.22, 0.24 | -0.00, -0.00, 0.01, 0.02, 0.02 | -0.00, -0.01, -0.00, -0.01, -0.01 | 0.00, 0.12, 0.34, 0.40, 0.41 | 0.00, 0.08, 0.13, 0.17, 0.16 |
| L4 keys 1 | -0.00, 0.10, 0.12, 0.12, 0.12 | -0.00, -0.01, -0.02, -0.01, -0.01 | -0.00, -0.01, -0.01, -0.00, -0.01 | 0.00, 0.04, 0.21, 0.23, 0.24 | 0.00, 0.07, 0.04, 0.02, 0.05 |
| L8 keys 0 | -0.00, 0.01, 0.02, 0.02, 0.02, 0.03, 0.02, 0.03, 0.03 | -0.00, 0.00, 0.01, 0.02, 0.02, 0.01, 0.01, -0.00, -0.00 | -0.00, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.00, -0.01 | 0.00, 0.11, 0.18, 0.22, 0.25, 0.25, 0.25, 0.24, 0.24 | 0.00, 0.05, 0.10, 0.16, 0.15, 0.15, 0.16, 0.16, 0.15 |

### Intervention: m = 4 copies, copies' residual after layer l replaced by the m = 1 state of the original token

If the copies' rewritten state carries the discount, the pull rises toward the independent-copies value (0.384) when patched early. Patching after the last layer is a no-op control.

| model | unpatched pull m=4 | patched after layer 0, 1, ... |
|---|---|---|
| L1 keys 0 | 0.106 | 0.11 |
| L1 keys 1 | 0.075 | 0.08 |
| L2 keys 0 | 0.087 | 0.16, 0.09 |
| L2 keys 1 | 0.063 | 0.17, 0.06 |
| L4 keys 0 | 0.080 | 0.15, 0.14, 0.11, 0.08 |
| L4 keys 1 | 0.067 | 0.15, 0.13, 0.09, 0.07 |
| L8 keys 0 | 0.093 | 0.11, 0.12, 0.11, 0.11, 0.10, 0.10, 0.10, 0.09 |

### Do the query's attention scores deduplicate copies beyond softmax arithmetic?

Pooled over heads per layer: (arith − observed) / (arith − q1) for the total mass on the copies, where arith = m·q1/(1+(m−1)·P1) is what identical scores give and q1 is what perfect score-based deduplication gives. 0 = no deduplication in the scores, 1 = full. Mean over seeds (range in brackets). Depth 1 is 0 by construction.

| model | m | layer 0, 1, ... trained | untrained (max abs) |
|---|---|---|---|
| L1 keys 0 | 2 | 0.00 [-0.00..0.00] | 0.00 |
| L1 keys 0 | 4 | 0.00 [-0.00..0.00] | 0.00 |
| L1 keys 1 | 2 | 0.00 [0.00..0.00] | 0.00 |
| L1 keys 1 | 4 | -0.00 [-0.00..0.00] | 0.00 |
| L2 keys 0 | 2 | 0.00 [0.00..0.00], 0.00 [0.00..0.00] | 0.00 |
| L2 keys 0 | 4 | -0.00 [-0.00..0.00], 0.01 [0.00..0.01] | 0.00 |
| L2 keys 1 | 2 | 0.00 [0.00..0.00], 0.41 [0.31..0.46] | 0.00 |
| L2 keys 1 | 4 | -0.00 [-0.00..-0.00], 0.43 [0.34..0.48] | 0.01 |
| L4 keys 0 | 2 | -0.00 [-0.00..0.00], 0.00 [-0.01..0.02], 0.00 [-0.02..0.02], 0.02 [0.02..0.04] | 0.00 |
| L4 keys 0 | 4 | -0.00 [-0.00..0.00], 0.01 [-0.01..0.06], 0.00 [-0.05..0.04], 0.06 [0.05..0.07] | 0.00 |
| L4 keys 1 | 2 | -0.00 [-0.00..0.00], 0.36 [0.29..0.50], 0.35 [0.28..0.42], 0.23 [0.14..0.38] | 0.00 |
| L4 keys 1 | 4 | -0.00 [-0.00..0.00], 0.40 [0.33..0.52], 0.39 [0.32..0.46], 0.25 [0.17..0.41] | 0.01 |
| L8 keys 0 | 2 | 0.00 [0.00..0.00], 0.00 [-0.01..0.01], -0.00 [-0.01..0.00], -0.00 [-0.01..0.00], 0.00 [-0.00..0.01], 0.09 [0.03..0.15], 0.02 [0.01..0.04], 0.05 [0.04..0.07] | 0.00 |
| L8 keys 0 | 4 | 0.00 [0.00..0.00], 0.02 [0.01..0.03], 0.00 [-0.01..0.02], 0.01 [-0.01..0.02], 0.02 [0.01..0.04], 0.12 [0.06..0.19], 0.05 [0.04..0.06], 0.07 [0.07..0.08] | 0.00 |

### Patching controls after layer 0 (reviewer-requested): which tokens' m=1 state removes the discount?

Pull toward the duplicated value; mean over seeds. 'random' = copies perturbed by a random vector of the same norm as the copies patch. Independent-copies reference: m=2 0.176, m=4 0.384; true duplicates 0.

| model | m | n | unpatched | copies ← m=1 | rest ← m=1 | all ← m=1 | random |
|---|---|---|---|---|---|---|---|
| L2 keys 0 | 2 | 3 | 0.048 | 0.068 | 0.074 | 0.094 | 0.048 |
| L2 keys 0 | 4 | 3 | 0.087 | 0.162 | 0.139 | 0.217 | 0.088 |
| L2 keys 1 | 2 | 3 | 0.030 | 0.076 | 0.056 | 0.102 | 0.026 |
| L2 keys 1 | 4 | 3 | 0.063 | 0.175 | 0.117 | 0.233 | 0.053 |
| L4 keys 0 | 2 | 3 | 0.046 | 0.062 | 0.066 | 0.082 | 0.046 |
| L4 keys 0 | 4 | 3 | 0.080 | 0.150 | 0.107 | 0.189 | 0.081 |
| L4 keys 1 | 2 | 3 | 0.034 | 0.063 | 0.049 | 0.077 | 0.034 |
| L4 keys 1 | 4 | 3 | 0.067 | 0.148 | 0.099 | 0.175 | 0.066 |
| L8 keys 0 | 2 | 2 | 0.044 | 0.048 | 0.047 | 0.052 | 0.043 |
| L8 keys 0 | 4 | 2 | 0.093 | 0.111 | 0.097 | 0.118 | 0.093 |
