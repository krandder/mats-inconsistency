### Shared fraction of the noise-free deviation across independently trained nets (8,000 test episodes)

| feature | same-recipe nets: mean over pairs [95% CI] (pair range) | cross-recipe pairs | E5 named-feature fraction, same scale (seeds 4 / 1 / 2) | reliability of one K = 64 estimate (range over nets) |
|---|---|---|---|---|
| mean | 0.48 [0.45, 0.51] (0.40 to 0.68) | 0.51 [0.48, 0.54] (0.37 to 0.70) | 0.16 / 0.06 / 0.12 | 0.980–0.994 |
| log-sd | 0.60 [0.58, 0.62] (0.54 to 0.76) | 0.59 [0.56, 0.62] (0.49 to 0.72) | 0.65 / 0.51 / 0.48 | 0.993–0.996 |
| KL | 0.48 [0.32, 0.63] (0.42 to 0.60) | 0.50 [0.35, 0.65] (0.36 to 0.68) | 0.17 / 0.21 / 0.17 | 0.983–1.001 |

Nets: L2_s0.05, L2seed1_s0.05, L2seed2_s0.05, L2ws0_s0.05, L2ws3_s0.05, L2mx0_s0.05, L2mx1_s0.05, L2mx4_s0.05; 10 same-recipe pairs, 15 cross-recipe pairs.

### Step 2: what carries the shared mean deviation

| features | mean: R² vs seed average (explained / shared) | log-sd: R² vs seed average (explained / shared) | KL: R² vs seed average (explained / shared) |
|---|---|---|---|
| named | 0.19 (0.24) | 0.73 (0.84) | 0.49 (0.61) |
| named + token-level | 0.40 (0.49) | 0.74 (0.84) | 0.34 (0.42) |

| description | mean: R² vs seed average / vs own deviation | log-sd: R² vs seed average / vs own deviation | KL: R² vs seed average / vs own deviation |
|---|---|---|---|
| L2_s0.05 DAG D1 | 0.28 / 0.77 | 0.53 / 0.82 | 0.24 / 0.64 |
| L2_s0.05 network itself (replicate 2) | 0.31 / 0.96 | 0.55 / 0.99 | 0.20 / 0.98 |
| L2_s0.05 E2 all-MLP-rank-8 (step 4, 12,896 parameters) | 0.25 / 0.92 | 0.50 / 0.96 | 0.16 / 0.89 |
| L2seed1_s0.05 DAG D1 | 0.31 / 0.61 | 0.52 / 0.66 | 0.23 / 0.36 |
| L2seed1_s0.05 network itself (replicate 2) | 0.24 / 0.96 | 0.48 / 0.99 | −0.12 / 0.99 |
| L2seed1_s0.05 E2 all-MLP-rank-8 (step 8, 12,896 parameters) | 0.22 / 0.92 | 0.41 / 0.95 | −0.24 / 0.93 |

### Step 3: the systematic spread (log-sd) error

Permutation importance (drop in held-out R² of the seed-averaged log-sd deviation): |Bayes mean| (scen) 0.604; Bayes log-sd 0.448; query-reading sd 0.198; full post s2 v2 0.086; full post s1 v2 0.085; full post s0 v2 0.076.

Compact model on |Bayes mean| (scen), Bayes log-sd, query-reading sd (28 parameters): R² vs the seed average 0.37 (explained / shared 0.43); all ~50 named features: 0.73 (0.84); per net: L2_s0.05 0.26, L2seed1_s0.05 0.18, L2seed2_s0.05 0.25, L2ws0_s0.05 0.27, L2ws3_s0.05 0.22.

- |Bayes mean| (scen): at its 5/25/50/75/95% quantiles 0.06, 0.29, 0.63, 1.07, 1.82 the predicted log-sd deviation is +0.026, +0.025, +0.027, +0.038, +0.078
- Bayes log-sd: at its 5/25/50/75/95% quantiles -1.52, -1.33, -1.15, -0.87, -0.55 the predicted log-sd deviation is +0.058, +0.038, +0.027, -0.006, -0.022
- query-reading sd: at its 5/25/50/75/95% quantiles 0.12, 0.31, 0.53, 0.86, 1.47 the predicted log-sd deviation is +0.015, +0.015, +0.027, +0.043, +0.059

Ridge probes in seed 4 (held-out R²): own log-sd deviation / seed-averaged log-sd deviation / |Bayes mean| (scen) / Bayes log-sd / query-reading sd

| site / view | own | seed average | |Bayes mean| (scen) | Bayes log-sd | query-reading sd |
|---|---|---|---|---|---|
| emb / query token | −0.00 | −0.00 | 0.00 | −0.00 | −0.00 |
| emb / source 0 tokens | 0.04 | 0.05 | 0.28 | 0.01 | 0.05 |
| emb / source 1 tokens | 0.02 | 0.05 | 0.27 | 0.01 | 0.05 |
| emb / source 2 tokens | 0.04 | 0.07 | 0.30 | 0.01 | 0.04 |
| emb / all tokens | 0.07 | 0.12 | 0.59 | 0.03 | 0.10 |
| a1 / query token | 0.03 | 0.07 | 0.74 | 0.23 | 0.81 |
| a1 / source 0 tokens | 0.05 | 0.07 | 0.38 | 0.22 | 0.10 |
| a1 / source 1 tokens | 0.04 | 0.07 | 0.36 | 0.21 | 0.10 |
| a1 / source 2 tokens | 0.06 | 0.09 | 0.38 | 0.23 | 0.08 |
| a1 / all tokens | 0.11 | 0.16 | 0.82 | 0.68 | 0.83 |
| m1 / query token | 0.08 | 0.11 | 0.76 | 0.26 | 0.84 |
| m1 / source 0 tokens | 0.08 | 0.11 | 0.49 | 0.26 | 0.11 |
| m1 / source 1 tokens | 0.06 | 0.10 | 0.47 | 0.25 | 0.13 |
| m1 / source 2 tokens | 0.09 | 0.12 | 0.49 | 0.27 | 0.11 |
| m1 / all tokens | 0.19 | 0.24 | 0.89 | 0.79 | 0.87 |
| r1 / query token | 0.09 | 0.12 | 0.76 | 0.24 | 0.81 |
| r1 / source 0 tokens | 0.07 | 0.11 | 0.49 | 0.26 | 0.11 |
| r1 / source 1 tokens | 0.06 | 0.10 | 0.48 | 0.26 | 0.13 |
| r1 / source 2 tokens | 0.09 | 0.12 | 0.49 | 0.27 | 0.11 |
| r1 / all tokens | 0.18 | 0.24 | 0.89 | 0.79 | 0.85 |
| a2 / query token | 0.13 | 0.21 | 0.95 | 0.90 | 0.76 |
| m2 / query token | 0.22 | 0.28 | 0.97 | 0.95 | 0.62 |
| r2 / query token | 0.21 | 0.29 | 0.94 | 0.94 | 0.44 |