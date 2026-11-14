# E1. Dictionaries in bits

Noise units = E||a - a_hat||^2 / (n s^2) per coordinate (below 1 = below the noise); bits = n/(2 ln 2) x noise units.
Best SAE = lowest held-out error over F in {1,2,4,8,16} x n at that k (raw TopK reconstruction; LS refit in brackets).


## arc_n64_s0.01_seed65

| layer | s | var/coord (noise units) | PR | eig > s^2 | k_PCA below noise | mean only | k=4: SAE best (refit) [F/n] / PCA / neuron | k=8: SAE best (refit) [F/n] / PCA / neuron | k=16: SAE best (refit) [F/n] / PCA / neuron | k=32: SAE best (refit) [F/n] / PCA / neuron | KT1 (L0<=n/4) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.01 | 6.9e+03 | 40.9 | 64 | 64 | 6.9e+03 | 2.7e+03 (2.6e+03) [16] / 5.7e+03 / 3.8e+03 | 1.8e+03 (1.7e+03) [2] / 4.7e+03 / 2.6e+03 | 536 (521) [2] / 3.2e+03 / 849 | 13.7 (13.3) [1] / 1.4e+03 / 9.18 | fires |
| 2 | 0.0098 | 5.4e+03 | 25.9 | 64 | 64 | 5.4e+03 | 1.7e+03 (1.6e+03) [16] / 3.9e+03 / 3.1e+03 | 1.1e+03 (1e+03) [16] / 2.9e+03 / 2.1e+03 | 453 (411) [4] / 1.7e+03 / 980 | 36.9 (32.9) [2] / 602 / 12.3 | fires |
| 4 | 0.01 | 3.2e+03 | 12.1 | 63 | 57 | 3.2e+03 | 629 (578) [16] / 1.7e+03 / 1.8e+03 | 401 (331) [16] / 1e+03 / 1.1e+03 | 139 (124) [4] / 440 / 426 | 10.6 (9.44) [4] / 122 / 5.99 | fires |
| 8 | 0.012 | 1.6e+03 | 3.3 | 55 | 45 | 1.6e+03 | 142 (127) [8] / 359 / 863 | 67.8 (52.4) [8] / 174 / 521 | 17.0 (15.9) [4] / 55.7 / 177 | 1.50 (1.22) [2] / 10.6 / 5.79 | fires |
| 16 | 0.019 | 1.1e+03 | 1.6 | 39 | 29 | 1.1e+03 | 24.6 (21.8) [4] / 64.7 / 553 | 10.3 (7.57) [8] / 25.1 / 326 | 2.61 (2.25) [4] / 5.97 / 102 | 0.50 (0.35) [8] / 0.50 / 1.00 | fires |
| 24 | 0.013 | 1.1e+03 | 1.4 | 38 | 24 | 1.1e+03 | 12.8 (11.1) [2] / 34.2 / 570 | 5.39 (4.12) [4] / 9.78 / 321 | 1.81 (1.53) [4] / 2.86 / 91.6 | 0.48 (0.33) [2] / 0.20 / 0.98 | fires |
| 31 | 0.012 | 1.1e+03 | 1.3 | 31 | 20 | 1.1e+03 | 9.42 (8.18) [4] / 20.5 / 518 | 4.29 (3.61) [4] / 7.29 / 276 | 1.49 (1.25) [4] / 1.80 / 55.7 | 0.36 (0.20) [2] / 0.0007 / -2.2e-13 | fires |

Error vs dictionary size at k = n/8 (raw, noise units), F/n = 1, 2, 4, 8, 16; and the 2x-epoch control at F = 4n:
- layer 1: 1.8e+03, 1.8e+03, 1.8e+03, 2e+03, 2e+03 (dead fraction at 16n: 0)
- layer 2: 1.3e+03, 1.2e+03, 1.2e+03, 1.2e+03, 1.1e+03 (dead fraction at 16n: 0)
- layer 4: 467, 438, 420, 410, 401 (dead fraction at 16n: 0.0088)
- layer 8: 76.9, 76.6, 70.5, 67.8, 67.8 (dead fraction at 16n: 0.49); 2x epochs F=4n: k=4: 143 -> 140, k=8: 70.5 -> 68.7, k=16: 17.0 -> 16.4, k=32: 1.73 -> 0.88
- layer 16: 11.5, 12.1, 10.7, 10.3, 11.1 (dead fraction at 16n: 0.74)
- layer 24: 6.10, 5.83, 5.39, 5.49, 5.49 (dead fraction at 16n: 0.57); 2x epochs F=4n: k=4: 12.9 -> 12.4, k=8: 5.39 -> 4.22, k=16: 1.81 -> 1.12, k=32: 0.53 -> 0.17
- layer 31: 4.32, 4.48, 4.29, 4.48, 5.03 (dead fraction at 16n: 0.36)

Active neurons (mean / max / ever active): L1: 32.0 / 48 / 64; L2: 33.2 / 48 / 64; L4: 33.4 / 43 / 64; L8: 33.8 / 43 / 59; L16: 33.7 / 38 / 46; L24: 33.6 / 37 / 45; L31: 27.4 / 32 / 38


# E2. One step: higher-order structure carried by latent-conditional Gaussians

All models carry the exact mean and full covariance. mean = nMSE of E[act(z')] (ARC units), cap3/cap4 = fraction of the per-neuron 3rd/4th cumulants of z' reproduced. TRUE_indep = independent fresh truth (sampling floor for independent samples; the models share latent and noise with the truth, so their floor is lower).


## arc_n64_s0.01_seed65

| layer | median abs g3 / g4 of z' | TRUE_indep | G (K=2) | EW3 oracle | EW4 oracle | k=4: DENSE / DENSEQ / best SPARSE [F/n] (mean nMSE; cap3) | k=8: DENSE / DENSEQ / best SPARSE [F/n] (mean nMSE; cap3) | k=16: DENSE / DENSEQ / best SPARSE [F/n] (mean nMSE; cap3) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.11 / 0.15 | 4.6e-06 | 1.0e-04 | 1.0e-05 | 2.7e-07 | 9.3e-05; 0.23 / 6.0e-05; 0.50 / 1.2e-05; 0.91 [16] | 7.6e-05; 0.37 / 3.5e-05; 0.72 / 4.7e-06; 0.97 [4] | 5.5e-05; 0.57 / 6.6e-06; 0.95 / 1.0e-06; 0.99 [4] |
| 2 | 0.18 / 0.30 | 7.9e-06 | 2.2e-04 | 2.4e-05 | 1.7e-06 | 1.2e-04; 0.62 / 9.3e-05; 0.75 / 1.3e-05; 0.96 [16] | 8.5e-05; 0.71 / 5.0e-05; 0.85 / 6.4e-06; 0.98 [8] | 4.2e-05; 0.86 / 8.5e-06; 0.98 / 2.2e-06; 0.99 [4] |
| 4 | 0.31 / 0.41 | 8.6e-06 | 2.1e-04 | 2.2e-05 | 2.3e-06 | 8.0e-05; 0.80 / 6.0e-05; 0.89 / 5.6e-06; 0.99 [16] | 4.4e-05; 0.89 / 1.8e-05; 0.97 / 3.9e-06; 1.00 [16] | 1.4e-05; 0.98 / 2.2e-06; 1.00 / 1.2e-06; 1.00 [4] |
| 8 | 0.58 / 0.65 | 1.2e-05 | 1.1e-04 | 2.6e-05 | 2.1e-06 | 3.4e-05; 0.97 / 1.6e-05; 0.99 / 3.5e-06; 1.00 [8] | 1.8e-05; 0.99 / 4.6e-06; 1.00 / 1.0e-06; 1.00 [16] | 4.1e-06; 1.00 / 4.0e-07; 1.00 / 1.9e-07; 1.00 [4] |
| 16 | 0.63 / 0.73 | 4.6e-06 | 9.0e-05 | 1.6e-05 | 2.4e-06 | 1.8e-05; 1.00 / 6.3e-06; 1.00 / 8.5e-07; 1.00 [8] | 8.5e-06; 1.00 / 9.9e-07; 1.00 / 2.8e-07; 1.00 [4] | 7.4e-07; 1.00 / 5.6e-08; 1.00 / 5.0e-08; 1.00 [16] |
| 24 | 0.60 / 0.64 | 9.0e-07 | 3.0e-05 | 6.3e-06 | 1.3e-06 | 1.1e-05; 1.00 / 3.7e-06; 1.00 / 3.8e-07; 1.00 [4] | 2.9e-06; 1.00 / 1.1e-07; 1.00 / 9.4e-08; 1.00 [16] | 1.9e-07; 1.00 / 1.2e-08; 1.00 / 1.4e-08; 1.00 [2] |
| 31 | 0.62 / 0.60 | 2.1e-06 | 4.2e-05 | 9.2e-06 | 2.0e-06 | 2.1e-06; 1.00 / 3.2e-07; 1.00 / 9.4e-08; 1.00 [8] | 8.5e-07; 1.00 / 3.5e-08; 1.00 / 1.9e-08; 1.00 [2] | 5.5e-08; 1.00 / 6.6e-09; 1.00 / 4.6e-09; 1.00 [4] |

Residual given the latent, per-neuron excess kurtosis (median [p90]): marginal / conditional-variance-normalised, in neuron space and projected by W_{l+1}; residual size in noise units:
- layer 1: k=4 SPARSE4n: 1.45 [1.56] / 355 [515]; proj 0.21 / 169; size 3e+03 | k=4 DENSE: 2.02 [2.30] / 129 [683]; proj 0.14 / 0.07 | k=8 SPARSE4n: 1.36 [1.69] / 84.4 [180]; proj 0.39 / 120; size 1.8e+03 | k=8 DENSE: 1.60 [2.00] / 208 [675]; proj 0.13 / 0.03 | k=16 SPARSE4n: 3.58 [4.67] / 82.5 [175]; proj 0.98 / 145; size 535 | k=16 DENSE: 0.98 [1.49] / 156 [313]; proj 0.10 / -0.0048
- layer 2: k=4 SPARSE4n: 1.53 [7.31] / 229 [375]; proj 0.36 / 229; size 1.9e+03 | k=4 DENSE: 1.71 [10.7] / 301 [1.9e+03]; proj 0.28 / 0.12 | k=8 SPARSE4n: 1.22 [7.65] / 154 [448]; proj 0.41 / 235; size 1.2e+03 | k=8 DENSE: 1.48 [10.5] / 253 [832]; proj 0.28 / 0.05 | k=16 SPARSE4n: 3.09 [11.4] / 67.9 [188]; proj 0.68 / 177; size 453 | k=16 DENSE: 0.97 [9.46] / 115 [316]; proj 0.29 / 0.72
- layer 4: k=4 SPARSE4n: 1.86 [86.0] / 183 [957]; proj 0.53 / 235; size 720 | k=4 DENSE: 2.35 [113] / 254 [2.8e+03]; proj 0.44 / 0.11 | k=8 SPARSE4n: 1.46 [80.1] / 216 [750]; proj 0.60 / 258; size 419 | k=8 DENSE: 1.81 [111] / 146 [1.1e+03]; proj 0.48 / 0.18 | k=16 SPARSE4n: 2.24 [105] / 119 [209]; proj 0.82 / 164; size 138 | k=16 DENSE: 0.97 [108] / 146 [607]; proj 0.58 / 302
- layer 8: k=4 SPARSE4n: 1.12 [423] / 234 [2e+03]; proj 0.93 / 218; size 143 | k=4 DENSE: 0.97 [491] / nan [nan]; proj 0.74 / 0.38 | k=8 SPARSE4n: 1.00 [554] / 182 [1e+03]; proj 0.88 / 197; size 70.3 | k=8 DENSE: 1.09 [488] / nan [nan]; proj 0.94 / 52.7 | k=16 SPARSE4n: 1.92 [487] / 75.3 [462]; proj 1.65 / 79.7; size 16.9 | k=16 DENSE: 1.71 [617] / nan [nan]; proj 1.53 / 173
- layer 16: k=4 SPARSE4n: 1.15 [7.4e+03] / 161 [383]; proj 1.11 / 180; size 24.6 | k=4 DENSE: 1.15 [7.5e+03] / nan [nan]; proj 1.00 / 0.31 | k=8 SPARSE4n: 1.10 [5.7e+03] / 189 [736]; proj 1.07 / 215; size 10.7 | k=8 DENSE: 1.28 [7.5e+03] / nan [nan]; proj 1.24 / 285 | k=16 SPARSE4n: 1.72 [1.3e+04] / 97.4 [289]; proj 1.66 / 148; size 2.53 | k=16 DENSE: 1.88 [2e+04] / nan [nan]; proj 1.80 / 95.7
- layer 24: k=4 SPARSE4n: 1.21 [3.1e+04] / 168 [507]; proj 1.21 / 233; size 12.8 | k=4 DENSE: 1.56 [6.5e+03] / nan [nan]; proj 1.37 / 64.3 | k=8 SPARSE4n: 0.99 [3.1e+04] / 259 [728]; proj 0.92 / 277; size 5.30 | k=8 DENSE: 1.89 [3.1e+04] / nan [nan]; proj 1.22 / 295 | k=16 SPARSE4n: 1.20 [3.1e+04] / 158 [455]; proj 1.11 / 224; size 1.67 | k=16 DENSE: 1.26 [3.1e+04] / nan [nan]; proj 1.21 / 258
- layer 31: k=4 SPARSE4n: 0.86 [9.8e+03] / 71.2 [453]; proj 0.83 / 217; size 9.41 | k=4 DENSE: 1.02 [9.9e+03] / nan [nan]; proj 0.95 / 26.0 | k=8 SPARSE4n: 0.58 [8.5e+03] / 26.3 [645]; proj 0.60 / 222; size 4.23 | k=8 DENSE: 0.75 [9.9e+03] / nan [nan]; proj 0.71 / 253 | k=16 SPARSE4n: 0.87 [7.7e+03] / 104 [516]; proj 0.78 / 310; size 1.32 | k=16 DENSE: 0.76 [9.9e+03] / nan [nan]; proj 0.99 / 391

**KT2** (sparse no better than dense at matched k in {n/16, n/8}; arc 0.01, layers 8-31): 0 of 8 cells -> does not fire
Cells (n, layer, k, best SPARSE, best dense, sparse >= dense): (64,8,4: 3.5e-06 vs 1.6e-05); (64,8,8: 1.0e-06 vs 4.6e-06); (64,16,4: 8.5e-07 vs 6.3e-06); (64,16,8: 2.8e-07 vs 9.9e-07); (64,24,4: 3.8e-07 vs 3.7e-06); (64,24,8: 9.4e-08 vs 1.1e-07); (64,31,4: 9.4e-08 vs 3.2e-07); (64,31,8: 1.9e-08 vs 3.5e-08)

**KT3** (best SPARSE k<=n/4 one-step mean nMSE >= G; arc 0.01, layers 8-31): 0 of 4 cells -> does not fire
Cells (n, layer: best SPARSE vs G): (64,8: 1.9e-07 vs 1.1e-04); (64,16: 5.0e-08 vs 9.0e-05); (64,24: 1.4e-08 vs 3.0e-05); (64,31: 4.6e-09 vs 4.2e-05)


# E3. Representation at a cut, pushed through the real layers (arc)

nMSE of E[a_L] (both Monte Carlo floors subtracted). TRUE = no replacement (pipeline control). K=2 / K3f = ARC's kprop end to end from the input (scored against the same truth).


## arc_n64_s0.01_seed65 (N3 = 1048576, truth 16777216)

| cut -> L | TRUE | G | k=4: DENSE / DENSEQ / best SPARSE [F/n] | k=8: DENSE / DENSEQ / best SPARSE [F/n] | k=16: DENSE / DENSEQ / best SPARSE [F/n] | k=32: DENSE / DENSEQ / best SPARSE [F/n] | K=2 e2e | K3f e2e |
|---|---|---|---|---|---|---|---|---|
| 4 -> 8 | -8.4e-08 | 6.2e-04 | 2.0e-04 / 1.4e-04 / 1.8e-05 [16] | 1.2e-04 / 7.8e-05 / 1.3e-05 [16] | 6.6e-05 / 4.6e-06 / 3.9e-06 [4] | 1.8e-05 / - / 9.3e-08 [4] | 3.8e-03 | 4.4e-04 |
| 8 -> 16 | -6.4e-07 | 4.3e-04 | 1.8e-04 / 8.5e-05 / 9.6e-06 [4] | 1.4e-04 / 3.8e-05 / 4.0e-06 [16] | 4.0e-05 / 6.3e-06 / 6.3e-07 [8] | 2.1e-06 / - / -5.8e-07 [2] | 6.7e-03 | 7.9e-04 |
| 8 -> 32 | -8.5e-07 | 1.3e-04 | 2.2e-05 / 2.0e-05 / -1.3e-07 [16] | 3.0e-05 / 1.0e-05 / 8.3e-07 [8] | 1.8e-05 / 4.8e-06 / 3.2e-08 [8] | 3.3e-06 / - / -9.3e-07 [8] | 1.1e-02 | 6.1e-02 |
| 16 -> 32 | 9.3e-07 | 3.6e-04 | 9.0e-05 / 2.2e-05 / 1.5e-06 [16] | 3.9e-05 / 8.2e-06 / 6.4e-07 [4] | 2.2e-06 / 2.1e-06 / -1.0e-07 [16] | 1.3e-06 / - / 3.8e-07 [2] | 1.1e-02 | 6.1e-02 |
| 24 -> 32 | -9.4e-07 | 2.6e-04 | 1.0e-04 / 3.3e-05 / 2.3e-06 [4] | 1.8e-05 / 3.6e-08 / -5.1e-07 [8] | 6.6e-07 / -8.4e-07 / -8.7e-07 [8] | -9.5e-07 / - / -9.7e-07 [4] | 1.1e-02 | 6.1e-02 |

**KT4** n=64: best SPARSE (k<=n/4) representation error at 32 over cuts 8/16/24 = -8.7e-07 vs K3f e2e 6.1e-02 -> does not fire (descriptive)


# E4. Transcoder chains (arc, sigma = 0.01)

