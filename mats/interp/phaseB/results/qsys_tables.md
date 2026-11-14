### Systematic fraction and learnability ceiling (held-out R² on 2,000 test episodes; fraction = R² / floor)

| model | floor mean / log-sd / KL | systematic R² mean / log-sd / KL (fraction) | ceiling R² mean / log-sd / KL (fraction) | regressor chosen (mean) |
|---|---|---|---|---|
| 2L seed 4 | 0.960 / 0.987 / 0.979 | 0.16 (0.16) / 0.64 (0.65) / 0.17 (0.17) | 0.03 (0.03) / 0.16 (0.16) / 0.12 (0.12) | HGB lr=0.1 leaves=63 l2=0.0 (val R2 0.187; MLP 0.186) |
| 2L seed 1 | 0.965 / 0.987 / 0.988 | 0.06 (0.06) / 0.51 (0.52) / 0.21 (0.21) | −0.00 (-0.00) / 0.11 (0.11) / 0.07 (0.07) | HGB lr=0.1 leaves=63 l2=1.0 (val R2 0.136; MLP 0.129) |
| 2L seed 2 | 0.982 / 0.989 / 0.989 | 0.12 (0.12) / 0.48 (0.48) / 0.17 (0.18) | −0.01 (-0.01) / 0.14 (0.14) / 0.01 (0.01) | MLP (val R2 0.124; best HGB 0.104) |
| 3L seed 13 | 0.948 / 0.983 / 0.987 | 0.10 (0.11) / 0.49 (0.50) / 0.14 (0.14) | −0.00 (-0.00) / 0.19 (0.20) / 0.06 (0.06) | MLP (val R2 0.134; best HGB 0.094) |

### Descriptions scored on the systematic part S(x): R² against S (raw) / R² of E[description | z] against S (projected, post hoc) / R² against the full deviation

**2L seed 4**

| description | mean | log-sd | KL |
|---|---|---|---|
| network itself (independent draws, seed 7) | −4.10 / −0.98 / 0.96 | 0.51 / 0.56 / 0.99 | −0.10 / 0.40 / 0.98 |
| DAG D1 (real weights) | −3.52 / −0.98 / 0.77 | 0.60 / 0.56 / 0.82 | 0.25 / 0.42 / 0.64 |
| scenario-mechanism regression | 0.05 / 0.04 / 0.01 | 0.29 / 0.26 / 0.17 | 0.47 / 0.43 / 0.29 |
| E3a head (h = 64) | −0.04 / −0.01 / 0.03 | 0.53 / 0.47 / 0.34 | 0.24 / 0.37 / 0.10 |
| E2 all-MLP-rank-8 (step 4, 12,896 parameters) | −4.39 / −1.06 / 0.92 | 0.46 / 0.56 / 0.96 | −0.21 / 0.33 / 0.89 |
| E2 cheapest (step 11, 12,512 parameters) | −5.13 / −1.63 / 0.49 | 0.37 / 0.51 / 0.78 | −0.57 / 0.25 / 0.26 |

**2L seed 1**

| description | mean | log-sd | KL |
|---|---|---|---|
| network itself (independent draws, seed 7) | −5.48 / −1.00 / 0.96 | 0.03 / 0.57 / 0.99 | −1.57 / 0.27 / 0.99 |
| DAG D1 (real weights) | −3.90 / −0.91 / 0.61 | 0.41 / 0.63 / 0.66 | 0.16 / 0.45 / 0.36 |
| scenario-mechanism regression | 0.02 / 0.02 / −0.00 | 0.34 / 0.32 / 0.16 | 0.56 / 0.51 / 0.19 |
| E3a head (h = 8) | 0.16 / 0.11 / 0.05 | 0.39 / 0.38 / 0.16 | 0.22 / 0.35 / 0.02 |
| E2 all-MLP-rank-8 (step 8, 12,896 parameters) | −5.74 / −1.06 / 0.92 | −0.04 / 0.54 / 0.95 | −2.08 / 0.22 / 0.93 |
| E2 cheapest (step 8, 12,896 parameters) | −5.74 / −1.06 / 0.92 | −0.04 / 0.54 / 0.95 | −2.08 / 0.22 / 0.93 |

**2L seed 2**

| description | mean | log-sd | KL |
|---|---|---|---|
| network itself (independent draws, seed 7) | −4.33 / −0.83 / 0.98 | −0.21 / 0.59 / 0.99 | −3.18 / −0.18 / 0.99 |
| DAG D1 (real weights) | −15 / −5.00 / −1.04 | −4.19 / −3.42 / −1.25 | −40 / −23 / −7.22 |
| DAG D1+lat (real weights) | −6.47 / −1.56 / 0.37 | −1.30 / 0.27 / 0.36 | −8.82 / −1.57 / −0.68 |
| scenario-mechanism regression | 0.01 / 0.01 / 0.00 | 0.38 / 0.36 / 0.17 | 0.47 / 0.45 / 0.13 |
| E3a head (h = 8) | 0.27 / 0.21 / 0.04 | 0.31 / 0.30 / 0.16 | 0.04 / 0.13 / 0.01 |
| E2 all-MLP-rank-8 (step 4, 12,896 parameters) | −5.49 / −1.28 / 0.85 | −0.38 / 0.51 / 0.92 | −4.35 / −0.51 / 0.85 |
| E2 cheapest (step 7, 9,814 parameters) | −18 / −11 / −2.07 | −12 / −10 / −4.76 | −176 / −131 / −34 |

**3L seed 13**

| description | mean | log-sd | KL |
|---|---|---|---|
| network itself (independent draws, seed 7) | −6.37 / −1.45 / 0.95 | 0.02 / 0.48 / 0.98 | −2.01 / −0.00 / 0.99 |
| DAG D1 (real weights) | −4.86 / −1.11 / 0.64 | 0.40 / 0.59 / 0.66 | −0.05 / 0.33 / 0.33 |
| scenario-mechanism regression | 0.00 / 0.00 / 0.00 | 0.34 / 0.33 / 0.16 | 0.32 / 0.29 / 0.09 |
| E3a head (h = 16) | −0.04 / −0.01 / 0.01 | 0.41 / 0.42 / 0.23 | −0.46 / −0.20 / −0.09 |
| E2 cheapest (step 10, 21,312 parameters) | −7.07 / −1.83 / 0.91 | 0.02 / 0.52 / 0.96 | −1.99 / −0.03 / 0.97 |
