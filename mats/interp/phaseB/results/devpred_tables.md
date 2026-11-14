### R² of predicted vs actual deviation (mean / log-sd / per-episode KL), 4,000 test episodes, K = 8

| model | deviation sd (mean / log-sd / KL) | noise floor | network = Bayes (0) | untrained net | best single-site surrogate (R² mean / log-sd / KL) [site:family] | MLP sites all LR16 | first attention site LRattn256 | cheap chain (LR4 + fixed pattern) |
|---|---|---|---|---|---|---|---|---|
| 2L seed 4 | 0.042 / 0.057 / 0.023 | 0.75 / 0.91 / 0.92 | -0.00 / -0.17 / -0.17 | -519 / -528 / -3281 | 0.75 / 0.91 / 0.91 [m1:LR16] | 0.74 / 0.90 / 0.91 | -27.30 / -31.46 / -143.90 | -75.87 / -27.68 / -1047.28 |
| 2L seed 1 | 0.040 / 0.055 / 0.024 | 0.74 / 0.89 / 0.93 | -0.00 / -0.18 / -0.15 | -552 / -567 / -3034 | 0.74 / 0.89 / 0.92 [m2:LR16] | 0.73 / 0.88 / 0.91 | -28.24 / -34.01 / -126.93 | -81.58 / -80.98 / -5005.10 |
| 2L seed 2 | 0.060 / 0.072 / 0.037 | 0.85 / 0.92 / 0.93 | -0.00 / -0.19 / -0.18 | -249 / -331 / -1273 | 0.86 / 0.92 / 0.93 [m1:LR16] | 0.83 / 0.89 / 0.88 | -10.08 / -15.90 / -44.34 | -178.16 / -32.83 / -3747.05 |
| 3L seed 13 | 0.037 / 0.052 / 0.023 | 0.66 / 0.87 / 0.93 | -0.00 / -0.22 / -0.12 | -721 / -892 / -4170 | 0.66 / 0.88 / 0.93 [m1:LR16] | 0.62 / 0.86 / 0.93 | -33.69 / -37.61 / -149.69 | -435.90 / -115.12 / -8574.43 |

### Single-site surrogates: R²(mean) per site and family (seed 4; other seeds in the JSON)

- emb: LR2 -60.71; LR4 -46.20; LR8 -41.32; LR16 0.75
- a1: drop -64.11; fixedpat -26.65; LRattn16 -33.77; LRattn64 -27.30; LRattn256 -27.30
- m1: LR2 -335.39; LR4 -2.21; LR8 0.74; LR16 0.75; TC64k4 -1.85; TC256k16 0.50
- r1: LR2 -166.31; LR4 -116.58; LR8 0.49; LR16 0.75
- a2: drop -319.34; fixedpat -7.09; LRattn16 -4.23; LRattn64 -4.07; LRattn256 -4.06
- m2: LR2 -16.54; LR4 -5.49; LR8 0.72; LR16 0.75; TC64k4 -0.35; TC256k16 0.64
- r2: LR2 -15.07; LR4 -5.65; LR8 -0.77; LR16 0.75

### Scenario predictions (fresh 32,768 episodes per seed); measured statistic and hit (Y/N)

| scenario | 2L seed 4 | 2L seed 1 | 2L seed 2 | 3L seed 13 |
|---|---|---|---|---|
| S1 | KL×2.74, dlogsd -0.003 (Y) | KL×2.83, dlogsd -0.004 (Y) | KL×2.47, dlogsd +0.011 (N) | KL×2.49, dlogsd -0.002 (Y) |
| S2 | KL×1.70, dlogsd +0.014 (N) | KL×1.69, dlogsd +0.020 (N) | KL×1.69, dlogsd +0.031 (N) | KL×1.44, dlogsd +0.019 (N) |
| S3 | RMS×1.85, KL×1.32 (Y) | RMS×1.86, KL×1.32 (Y) | RMS×1.88, KL×1.54 (Y) | RMS×1.87, KL×1.43 (Y) |
| S4 | KL×0.88 (N) | KL×0.93 (N) | KL×0.74 (N) | KL×0.92 (N) |
| S5 | KL×1.14 (N) | KL×1.18 (N) | KL×1.05 (N) | KL×1.26 (Y) |
| S6 | RMS×1.17 (Y) | RMS×1.20 (Y) | RMS×1.11 (Y) | RMS×1.22 (Y) |
| S7 | agree 0.53 (base 0.52) (N) | agree 0.53 (base 0.52) (N) | agree 0.64 (base 0.60) (Y) | agree 0.52 (base 0.51) (N) |
| S8 | signed -0.066, RMS×3.50 (Y) | signed -0.068, RMS×3.59 (Y) | signed -0.090, RMS×3.03 (Y) | signed -0.080, RMS×3.87 (Y) |
| S9 | KL×2.50 (Y) | KL×2.52 (Y) | KL×2.03 (Y) | KL×2.03 (Y) |

### Lateral arm: sparse surrogate DAGs (KL over the DAG, deviation R², treewidth)

| model | variant | KL (nats) | R² mean / log-sd / KL | scenario hit agreement with the network | treewidth | largest clique dim | attention edges |
|---|---|---|---|---|---|---|---|
| 2L seed 4 | exact | 0.0 | noise floor 0.75 / 0.91 / 0.92 | nan | 30 | 992 | 482 |
| 2L seed 4 | D1 | 302.0 | 0.60 / 0.74 / 0.75 | 0.89 | 20 | 672 | 332 |
| 2L seed 4 | D1+lat | 301.2 | 0.60 / 0.74 / 0.75 | 0.89 | 21 | 704 | 338 |
| 2L seed 4 | D2 | 307.9 | 0.52 / 0.72 / 0.72 | 0.89 | 16 | 544 | 320 |
| 2L seed 4 | D2+lat | 307.1 | 0.53 / 0.72 / 0.72 | 0.89 | 18 | 608 | 326 |
| 2L seed 4 | D3 | 367.6 | -0.22 / -2.34 / -0.35 | 0.67 | 5 | 192 | 308 |
| 2L seed 4 | D3+lat | 366.8 | -0.22 / -2.34 / -0.34 | 0.67 | 7 | 256 | 314 |
| 2L seed 1 | exact | 0.0 | noise floor 0.74 / 0.89 / 0.93 | nan | 30 | 992 | 482 |
| 2L seed 1 | D1 | 232.8 | 0.52 / 0.65 / 0.64 | 0.89 | 20 | 672 | 332 |
| 2L seed 1 | D1+lat | 231.8 | 0.52 / 0.65 / 0.64 | 0.89 | 21 | 704 | 338 |
| 2L seed 1 | D2 | 324.3 | 0.45 / 0.61 / 0.59 | 0.89 | 16 | 544 | 320 |
| 2L seed 1 | D2+lat | 323.2 | 0.46 / 0.61 / 0.59 | 0.89 | 18 | 608 | 326 |
| 2L seed 1 | D3 | 407.1 | -0.01 / -3.34 / -2.46 | 0.67 | 5 | 192 | 308 |
| 2L seed 1 | D3+lat | 406.1 | -0.01 / -3.33 / -2.45 | 0.67 | 7 | 256 | 314 |
| 2L seed 2 | exact | 0.0 | noise floor 0.85 / 0.92 / 0.93 | nan | 30 | 992 | 482 |
| 2L seed 2 | D1 | 4141.9 | -0.99 / -1.02 / -4.73 | 0.44 | 20 | 672 | 332 |
| 2L seed 2 | D1+lat | 2831.1 | 0.28 / 0.31 / -0.75 | 0.56 | 21 | 704 | 338 |
| 2L seed 2 | D2 | 6385.6 | -1.31 / -3.61 / -9.19 | 0.44 | 16 | 544 | 320 |
| 2L seed 2 | D2+lat | 5074.8 | -0.32 / -0.63 / -2.57 | 0.44 | 18 | 608 | 326 |
| 2L seed 2 | D3 | 7171.5 | -2.74 / -23.36 / -56.33 | 0.33 | 5 | 192 | 308 |
| 2L seed 2 | D3+lat | 5860.7 | -5.38 / -11.82 / -51.09 | 0.56 | 7 | 256 | 314 |
| 3L seed 13 | exact | 0.0 | noise floor 0.66 / 0.87 / 0.93 | nan | 30 | 992 | 723 |
| 3L seed 13 | D1 | 232.6 | 0.45 / 0.63 / 0.72 | 0.78 | 30 | 992 | 573 |
| 3L seed 13 | D1+lat | 231.7 | 0.45 / 0.64 / 0.72 | 0.78 | 30 | 992 | 579 |
| 3L seed 13 | D1-L2src | 5428.9 | -2.18 / -7.17 / -24.92 | 0.67 | 20 | 672 | 423 |
| 3L seed 13 | D1-L2lat | 3186.6 | -1.08 / -0.13 / -0.12 | 0.44 | 21 | 704 | 429 |
| 3L seed 13 | D1+lat-L2lat | 3185.7 | -1.08 / -0.12 / -0.12 | 0.44 | 21 | 704 | 435 |
