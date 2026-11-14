### E8.1 Linearisation constant (fit episodes, all-net average deviation)

| c | R² (all-net average) | mean of per-net R² |
|---|---|---|
| c = 0 | −6.52 | −4.56 |
| c = Bayes mean | −1.56 | −1.12 |

Chosen: c = Bayes mean.

### E8.1 Source mis-weighting vs the seed-averaged test mean deviation (finite-ensemble scale; 0.5 = explains)

| nets | measured-δ ceiling: R² / score (4,000 ep.) | compact model: R² / score [95% CI] (8,000 ep.) | verdict | compact score, same 4,000 ep. | weight-error R² of g (test) |
|---|---|---|---|---|---|
| all | −1.81 / −1.98 | −0.66 / −0.73 [−0.96, −0.55] | does not explain | −0.68 | 0.18 |
| A | −1.93 / −2.10 | −0.79 / −0.86 [−1.07, −0.69] | does not explain | −0.81 | 0.18 |
| B | −1.31 / −1.53 | −0.39 / −0.45 [−0.63, −0.32] | does not explain | −0.40 | 0.15 |

### E8.1 Compact model with one input dropped (score; informative)

| input dropped | all | A | B |
|---|---|---|---|
| log own labelled SS | −0.61 | −0.60 | −0.45 |
| log other SS (smaller) | −0.58 | −0.56 | −0.44 |
| log other SS (larger) | −0.63 | −0.74 | −0.40 |
| Bayes weight W_s | 0.08 | 0.05 | 0.11 |
| |o*_s - mean of other o*| | −0.42 | −0.55 | −0.23 |
| max |ID cosine| | −0.72 | −0.86 | −0.45 |

### E8.1b (post hoc) Exact path decomposition of the mean deviation: dev(o) = dev(agreement point) + Σ_s (o*_s − μ_B) d̄_s

| nets | agreement offset: R² / score | reading combination: R² / score | both (K = 8 total): R² / score | compact on d̄: score [95% CI] | compact fitted directly: score [95% CI] |
|---|---|---|---|---|---|
| all | 0.12 / 0.13 | 0.85 / 0.93 | 0.98 / 1.08 | 0.03 [−0.01, 0.07] (does not explain) | 0.18 [0.15, 0.21] (does not explain) |
| A | 0.07 / 0.08 | 0.84 / 0.92 | 0.96 / 1.05 | 0.00 [−0.04, 0.05] (does not explain) | 0.16 [0.12, 0.19] (does not explain) |
| B | 0.14 / 0.16 | 0.84 / 0.98 | 0.98 / 1.14 | 0.07 [0.03, 0.11] (does not explain) | 0.19 [0.16, 0.22] (does not explain) |

Held-out nb6: reference R² (other five A nets' average) 0.55; compact on d̄ R² 0.01 (ratio 0.01); compact direct R² 0.11 (ratio 0.19); holds if ratio ≥ 0.5.

| input dropped (score) | on d̄: all / A / B | direct: all / A / B |
|---|---|---|
| log own labelled SS | 0.09 / 0.06 / 0.11 | 0.11 / 0.10 / 0.13 |
| log other SS (smaller) | 0.05 / 0.03 / 0.08 | 0.18 / 0.15 / 0.18 |
| log other SS (larger) | 0.04 / 0.01 / 0.08 | 0.18 / 0.16 / 0.19 |
| Bayes weight W_s | −0.02 / −0.07 / 0.03 | 0.14 / 0.11 / 0.16 |
| |o*_s - mean of other o*| | −0.02 / −0.07 / 0.04 | 0.14 / 0.11 / 0.17 |
| max |ID cosine| | 0.04 / 0.01 / 0.07 | 0.18 / 0.16 / 0.19 |

Partial dependence of g (group A, direct fit): g at the 10th / 50th / 90th percentile of one input, others at medians

| input | values | g |
|---|---|---|
| log own labelled SS | −1.57 / 0.40 / 2.27 | -0.021 / -0.016 / -0.004 |
| log other SS (smaller) | −2.03 / −0.62 / 1.18 | -0.014 / -0.016 / -0.012 |
| log other SS (larger) | −0.53 / 1.30 / 2.58 | -0.016 / -0.016 / -0.019 |
| Bayes weight W_s | 0.05 / 0.23 / 0.64 | 0.006 / -0.016 / -0.030 |
| |o*_s - mean of other o*| | 0.11 / 0.64 / 1.84 | -0.019 / -0.016 / -0.023 |
| max |ID cosine| | 0.11 / 0.27 / 0.48 | -0.016 / -0.016 / -0.014 |

### E8.2 Layer-2 query-row patches: shared mean error retained (1 = unchanged, 0 = removed) and mean squared deviation ratio

| patch | A: retained [95% CI] | A: range over nets | A: MSD ratio | B: retained [95% CI] | B: range | B: MSD ratio | A / B retained vs all other nets |
|---|---|---|---|---|---|---|---|
| Pbayes | 0.24 [0.02, 0.44] | 0.20 to 0.27 | 6.83 | 0.43 [0.25, 0.59] | 0.34 to 0.48 | 3.68 | 0.36 / 0.35 |
| PbayesQQ | 0.33 [0.12, 0.52] | 0.25 to 0.45 | 5.93 | 0.78 [0.67, 0.88] | 0.68 to 0.88 | 2.17 | 0.54 / 0.69 |
| Ppost | 0.29 [0.12, 0.44] | 0.24 to 0.35 | 3.31 | −0.06 [−0.23, 0.11] | −0.15 to 0.06 | 2.04 | −0.07 / 0.07 |
| PpostQQ | 0.38 [0.25, 0.50] | 0.28 to 0.49 | 2.05 | 0.51 [0.44, 0.58] | 0.38 to 0.61 | 0.82 | 0.14 / 0.54 |
| Ptrue | 0.47 [0.22, 0.71] | 0.43 to 0.52 | 11.22 | 0.09 [−0.10, 0.27] | −0.01 to 0.17 | 5.79 | 0.14 / 0.23 |
| Puniform | 1.38 [1.25, 1.54] | 1.29 to 1.52 | 8.14 | 1.68 [1.54, 1.85] | 1.54 to 1.84 | 4.57 | 1.90 / 1.57 |
| Pmean | 1.60 [1.43, 1.77] | 1.44 to 1.84 | 12.69 | 2.06 [1.87, 2.28] | 1.74 to 2.32 | 8.43 | 2.26 / 1.92 |

### E8.2 Layer-2 query-row patches: shared spread (log-sd) error retained (1 = unchanged, 0 = removed) and mean squared deviation ratio

| patch | A: retained [95% CI] | A: range over nets | A: MSD ratio | B: retained [95% CI] | B: range | B: MSD ratio | A / B retained vs all other nets |
|---|---|---|---|---|---|---|---|
| Pbayes | 1.10 [0.97, 1.21] | 1.05 to 1.13 | 10.56 | 0.66 [0.54, 0.78] | 0.61 to 0.73 | 6.96 | 0.90 / 0.87 |
| PbayesQQ | 1.08 [0.98, 1.18] | 1.05 to 1.12 | 6.82 | 0.98 [0.92, 1.04] | 0.93 to 1.10 | 2.19 | 0.98 / 1.04 |
| Ppost | 1.07 [0.96, 1.16] | 0.98 to 1.12 | 9.02 | 0.59 [0.47, 0.71] | 0.53 to 0.65 | 5.92 | 0.86 / 0.80 |
| PpostQQ | 1.06 [0.96, 1.15] | 1.01 to 1.11 | 5.82 | 0.91 [0.86, 0.95] | 0.87 to 0.99 | 1.54 | 0.96 / 0.96 |
| Ptrue | 1.15 [1.02, 1.26] | 1.08 to 1.18 | 11.30 | 0.69 [0.56, 0.80] | 0.64 to 0.75 | 7.23 | 0.96 / 0.90 |
| Puniform | 1.07 [0.98, 1.18] | 0.98 to 1.19 | 9.29 | 1.28 [1.21, 1.36] | 1.18 to 1.41 | 4.83 | 1.36 / 1.18 |
| Pmean | 1.18 [1.03, 1.36] | 1.10 to 1.32 | 10.10 | 1.64 [1.52, 1.76] | 1.48 to 1.95 | 5.77 | 1.50 / 1.55 |

### E8.2 Mean KL(Bayes‖net) ratio, patched / base (informative)

| patch | A | B |
|---|---|---|
| Pbayes | 9.72 | 6.25 |
| PbayesQQ | 7.30 | 2.68 |
| Ppost | 5.85 | 3.98 |
| PpostQQ | 3.75 | 1.29 |
| Ptrue | 13.28 | 8.01 |
| Puniform | 8.09 | 4.48 |
| Pmean | 11.58 | 7.81 |

### E8.2 Per net (retained shared error vs same-group others; mean / log-sd)

| net | group | Pbayes | Ppost | Ptrue | Puniform | Pmean |
|---|---|---|---|---|---|---|
| L2_s0.05 | A | 0.25 / 1.12 | 0.26 / 1.12 | 0.44 / 1.18 | 1.40 / 1.06 | 1.50 / 1.10 |
| L2seed1_s0.05 | A | 0.20 / 1.09 | 0.24 / 1.08 | 0.43 / 1.16 | 1.52 / 1.05 | 1.68 / 1.12 |
| L2mx0_s0.05 | A | 0.27 / 1.08 | 0.35 / 1.06 | 0.52 / 1.13 | 1.29 / 1.04 | 1.44 / 1.10 |
| L2mx1_s0.05 | A | 0.25 / 1.13 | 0.31 / 1.06 | 0.46 / 1.16 | 1.34 / 1.09 | 1.50 / 1.32 |
| L2mx2_s0.05 | A | 0.26 / 1.13 | 0.34 / 1.11 | 0.50 / 1.17 | 1.33 / 1.19 | 1.62 / 1.28 |
| L2nb6_s0.05 | A | 0.24 / 1.05 | 0.24 / 0.98 | 0.44 / 1.08 | 1.42 / 0.98 | 1.84 / 1.19 |
| L2seed2_s0.05 | B | 0.38 / 0.73 | −0.15 / 0.62 | 0.01 / 0.75 | 1.78 / 1.41 | 2.32 / 1.95 |
| L2ws0_s0.05 | B | 0.46 / 0.70 | 0.06 / 0.65 | 0.17 / 0.72 | 1.54 / 1.21 | 1.74 / 1.48 |
| L2ws3_s0.05 | B | 0.44 / 0.64 | −0.07 / 0.57 | 0.04 / 0.66 | 1.55 / 1.18 | 1.92 / 1.51 |
| L2mx4_s0.05 | B | 0.46 / 0.64 | −0.02 / 0.58 | 0.15 / 0.69 | 1.79 / 1.27 | 1.99 / 1.61 |
| L2nb5_s0.05 | B | 0.34 / 0.67 | −0.13 / 0.60 | −0.01 / 0.69 | 1.59 / 1.30 | 2.27 / 1.67 |
| L2nb7_s0.05 | B | 0.48 / 0.61 | −0.04 / 0.53 | 0.15 / 0.64 | 1.84 / 1.32 | 2.12 / 1.63 |

### E8.2b (post hoc) Dose-response and single-head Bayes patches: shared mean error retained and MSD ratio

| patch | A: retained [95% CI] | A: MSD ratio | B: retained [95% CI] | B: MSD ratio |
|---|---|---|---|---|
| PbA10 | 0.89 [0.86, 0.92] | 0.94 | 0.93 [0.90, 0.95] | 0.93 |
| PbA25 | 0.75 [0.67, 0.82] | 1.28 | 0.82 [0.76, 0.88] | 1.09 |
| PbA50 | 0.56 [0.40, 0.68] | 2.59 | 0.67 [0.57, 0.77] | 1.75 |
| PbH | 0.56 [0.41, 0.68] | 3.55 | 0.61 [0.49, 0.71] | 2.46 |
| PbH50 | 0.75 [0.67, 0.83] | 1.58 | 0.78 [0.71, 0.84] | 1.29 |

Cause share φ = (1 − retained)/a: A: a = 0.1 1.06, a = 0.25 0.99; B: a = 0.1 0.74, a = 0.25 0.71

### E8.2b (post hoc) Dose-response and single-head Bayes patches: shared spread (log-sd) error retained and MSD ratio

| patch | A: retained [95% CI] | A: MSD ratio | B: retained [95% CI] | B: MSD ratio |
|---|---|---|---|---|
| PbA10 | 1.00 [0.98, 1.01] | 0.88 | 0.95 [0.94, 0.97] | 0.81 |
| PbA25 | 1.00 [0.96, 1.03] | 1.22 | 0.89 [0.85, 0.92] | 0.96 |
| PbA50 | 1.02 [0.95, 1.08] | 3.11 | 0.79 [0.71, 0.86] | 2.20 |
| PbH | 1.02 [0.95, 1.08] | 3.55 | 0.80 [0.72, 0.87] | 2.84 |
| PbH50 | 0.99 [0.95, 1.03] | 1.50 | 0.88 [0.83, 0.92] | 1.17 |

Cause share φ = (1 − retained)/a: A: a = 0.1 0.02, a = 0.25 0.00; B: a = 0.1 0.47, a = 0.25 0.45

### E8.4 (post hoc) Attention value-weighting model: dev = k × Σ_s (q_s − p_s) o*_s (one parameter k, fitted on fit episodes)

| model | all: k / score [95% CI] | A: k / score [95% CI] | B: k / score [95% CI] | held-out nb6: R² (ratio to 0.55 reference) |
|---|---|---|---|---|
| M1 (QQ masses vs posterior share) | 0.08 / 0.14 [0.12, 0.16] | 0.06 / 0.08 [0.06, 0.10] | 0.10 / 0.16 [0.13, 0.18] | 0.06 (0.10) |
| M0 flat control (1/3 vs posterior share) | 0.02 / 0.06 [0.04, 0.07] | 0.01 / 0.02 [0.01, 0.03] | 0.03 / 0.08 [0.06, 0.09] | 0.00 (0.01) |
| M1s (source masses vs posterior share) | 0.05 / 0.09 [0.07, 0.11] | 0.04 / 0.05 [0.03, 0.06] | 0.05 / 0.11 [0.09, 0.13] | 0.02 (0.04) |
| M1w (QQ masses vs clipped W_s) | 0.04 / 0.05 [0.03, 0.07] | 0.04 / 0.05 [0.04, 0.07] | 0.04 / 0.04 [0.02, 0.06] | 0.04 (0.07) |

Flatness (fit episodes): A: slope of q on p 0.67, corr 0.93, mean entropy q 0.97 vs p 0.86; B: slope of q on p 0.58, corr 0.84, mean entropy q 0.98 vs p 0.86

### E8.3 Held-out net

Triggered by patch. nb6: compact R² −0.52 vs other five A nets' average R² 0.55 (ratio −0.94; holds if ≥ 0.5: False).
 Pbayes on nb6 retains 0.24 (mean) / 1.05 (log-sd).

### Predictions

- P68: fails (value −1.81)
- P69: fails (A −0.86, B −0.45)
- P70: fails (Pbayes_mean 0.24, Pbayes_spread 1.10, Puniform_msd 8.14, Pmean_msd 12.69)
- P72: holds (combination 0.93, offset 0.13)
- P73: fails (on_dbar 0.00, direct 0.16)
- P74: fails (ratio 0.19)
- P75: holds (phi_a10 1.06, phi_a25 0.99, msd_a10 0.94)
- P76: fails (M1 0.08, M0 0.02)
- P77: fails (ratio 0.10)
- P78: holds (slope 0.67)
- P71: holds ()
